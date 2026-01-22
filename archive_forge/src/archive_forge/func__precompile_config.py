import builtins
import copy
import functools
import hashlib
import inspect
import json
import logging
import math
import operator
import os
import os.path
import re
import threading
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch
import torch.autograd.profiler as autograd_profiler
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import dynamo_timed
from torch.utils._triton import has_triton, has_triton_package
from . import config
from .codecache import cache_dir, CudaKernelParamCache
from .coordinate_descent_tuner import CoordescTuner
from .ir import ReductionHint, TileHint
from .utils import (
def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: Optional[int]):
    """Ahead of time compile a given autotuner config."""
    compile_meta = copy.deepcopy(self.triton_meta)
    for k, v in cfg.kwargs.items():
        compile_meta['constants'][self.fn.arg_names.index(k)] = v
    compile_meta['num_warps'] = cfg.num_warps
    compile_meta['num_stages'] = cfg.num_stages
    compile_meta['debug'] = config.assert_indirect_indexing and torch.version.hip is None
    compile_meta['device_type'] = 'cuda' if torch.version.hip is None else 'hip'
    if warm_cache_only_with_cc:
        return (triton.compile(self.fn, warm_cache_only=True, cc=warm_cache_only_with_cc, **compile_meta), None)
    with torch.cuda.device(compile_meta['device']):
        torch.cuda.synchronize(torch.cuda.current_device())
        binary = triton.compile(self.fn, **compile_meta)
        binary._init_handles()
    call_args = [arg for i, arg in enumerate(self.fn.arg_names) if i not in self.fn.constexprs]
    def_args = [name for name in self.fn.arg_names if name not in cfg.kwargs]
    scope = {'grid_meta': cfg.kwargs, 'bin': binary, 'torch': torch, 'set_device': torch.cuda.set_device, 'current_device': torch.cuda.current_device}
    exec(f'\n            def launcher({', '.join(def_args)}, grid, stream):\n                if callable(grid):\n                    grid_0, grid_1, grid_2 = grid(grid_meta)\n                else:\n                    grid_0, grid_1, grid_2 = grid\n\n                if hasattr(bin, "num_ctas"):\n                    bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps,\n                                bin.num_ctas, *bin.clusterDims, bin.shared,\n                                stream, bin.cu_function, None, None, None,\n                                {', '.join(call_args)})\n                else:\n                    bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared,\n                                stream, bin.cu_function, None, None, None,\n                                {', '.join(call_args)})\n                return bin\n            '.lstrip(), scope)
    launcher = scope['launcher']
    launcher.config = cfg
    launcher.n_regs = getattr(binary, 'n_regs', None)
    launcher.n_spills = getattr(binary, 'n_spills', None)
    launcher.shared = getattr(binary, 'shared', None)
    launcher.store_cubin = config.triton.store_cubin
    if launcher.store_cubin:
        launcher.fn = self.fn
        launcher.bin = binary
    return (binary, launcher)