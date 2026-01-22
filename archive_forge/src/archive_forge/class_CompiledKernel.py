from __future__ import annotations
import functools
import hashlib
import json
import os
import re
from collections import namedtuple
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from .._C.libtriton.triton import (ClusterInfo, TMAInfos, add_external_libs, compile_ptx_to_cubin, get_env_vars,
from ..common.backend import get_backend, get_cuda_version_key, path_to_ptxas
from ..common.build import is_hip
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from ..runtime.driver import driver
from ..runtime.jit import (JITFunction, get_cuda_stream, get_current_device, get_device_capability)
from ..tools.disasm import get_sass
from .code_generator import ast_to_ttir
from .make_launcher import make_stub
from .utils import (InfoFromBackendForTensorMap, TensorMapManager, get_ids_of_tensormaps, parse_tma_info)
class CompiledKernel:
    launch_enter_hook = None
    launch_exit_hook = None
    tensormap_manager = TensorMapManager()

    def __init__(self, fn, so_path, metadata, asm):
        import importlib.util
        spec = importlib.util.spec_from_file_location('__triton_launcher', so_path)
        mod = importlib.util.module_from_spec(spec)
        self.fn = fn
        spec.loader.exec_module(mod)
        self.c_wrapper = getattr(mod, 'launch')
        self.shared = metadata['shared']
        self.num_warps = metadata['num_warps']
        if 'threads_per_warp' in metadata:
            self.threads_per_warp = metadata['threads_per_warp']
        self.num_ctas = metadata['num_ctas']
        self.num_stages = metadata['num_stages']
        self.clusterDims = metadata['clusterDims']
        if 'tensormaps_info' in metadata:
            self.tensormaps_info = metadata['tensormaps_info']
        self.constants = metadata['constants']
        self.device_type = metadata['device_type']
        self.device_backend = get_backend(self.device_type) if self.device_type not in ['cuda'] else None
        self.asm = asm
        self.metadata = metadata
        self.cu_module = None
        self.cu_function = None

    def _init_handles(self):
        if self.cu_module is not None:
            return
        if self.device_type in ['cuda']:
            device = get_current_device()
            bin_path = {driver.HIP: 'hsaco_path', driver.CUDA: 'cubin'}[driver.backend]
            max_shared = driver.utils.get_device_properties(device)['max_shared_mem']
            fn_load_binary = driver.utils.load_binary
        else:
            assert self.device_backend
            device = self.device_backend.get_current_device()
            bin_path = self.device_backend.get_kernel_bin()
            max_shared = self.device_backend.get_device_properties(device)['max_shared_mem']
            fn_load_binary = self.device_backend.get_load_binary_fn()
        if self.shared > max_shared:
            raise OutOfResources(self.shared, max_shared, 'shared memory')
        mod, func, n_regs, n_spills = fn_load_binary(self.metadata['name'], self.asm[bin_path], self.shared, device)
        self.n_spills = n_spills
        self.n_regs = n_regs
        self.cu_module = mod
        self.cu_function = func

    def __getattribute__(self, name):
        if name == 'c_wrapper':
            self._init_handles()
        return super().__getattribute__(name)

    def assemble_tensormap_to_arg(self, args):
        args_with_tma = list(args)
        if hasattr(self, 'tensormaps_info'):
            args_ptr = tuple([arg.data_ptr() if hasattr(arg, 'data_ptr') else arg for arg in args])
            for i, e in enumerate(self.tensormaps_info):
                args_with_tma.append(CompiledKernel.tensormap_manager[e, args_ptr])
        return args_with_tma

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            args_expand = self.assemble_tensormap_to_arg(args)
            if stream is None:
                if self.device_type in ['cuda']:
                    stream = get_cuda_stream()
                else:
                    stream = get_backend(self.device_type).get_stream(None)
            self.c_wrapper(grid[0], grid[1], grid[2], self.num_warps, self.num_ctas, self.clusterDims[0], self.clusterDims[1], self.clusterDims[2], self.shared, stream, self.cu_function, CompiledKernel.launch_enter_hook, CompiledKernel.launch_exit_hook, self, *args_expand)
        return runner