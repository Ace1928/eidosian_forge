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
class CachingAutotuner(KernelInterface):
    """
    Simplified version of Triton autotuner that has no invalidation
    key and caches the best config to disk to improve cold start times.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(self, fn, triton_meta, configs, save_cache_hook, mutated_arg_names, heuristic_type, size_hints=None, inductor_meta=None):
        super().__init__()
        self.fn = fn
        self.triton_meta = triton_meta
        self.inductor_meta = {} if inductor_meta is None else inductor_meta
        self.save_cache_hook = save_cache_hook
        self.mutated_arg_names = mutated_arg_names
        self.configs = configs
        self.heuristic_type = heuristic_type
        if log.isEnabledFor(logging.DEBUG):
            log.debug('CachingAutotuner gets %d configs', len(self.configs))
            for c in self.configs:
                log.debug(c)
        self.launchers = []
        self.lock = threading.Lock()
        if os.getenv('TRITON_CACHE_DIR') is None:
            os.environ['TRITON_CACHE_DIR'] = os.path.join(cache_dir(), 'triton', str(self.triton_meta.get('device', 0)))
        self.size_hints = size_hints
        self.coordesc_tuner = CoordescTuner(is_mm=False, name=self.fn.__name__, size_hints=size_hints)
        self.record_function_ctx = torch._C._profiler._RecordFunctionFast(self.inductor_meta.get('kernel_name', 'triton kernel'))

    def precompile(self, warm_cache_only_with_cc=None):
        with self.lock:
            if self.launchers:
                return
            self.launchers = []
            compiled_binaries = []
            for c in self.configs:
                try:
                    compiled_binary, launcher = self._precompile_config(c, warm_cache_only_with_cc)
                except OutOfResources:
                    continue
                self.launchers.append(launcher)
                compiled_binaries.append(compiled_binary)
            if len(self.launchers) == 0:
                raise RuntimeError('No valid triton configs. Report a fatal compilation error')
            seen_configs = set(self.configs)
            device_interface = get_interface_for_device('cuda')
            device_prop = device_interface.Worker.get_device_properties(self.triton_meta['device'])
            if config.dynamic_scale_rblock and self.heuristic_type == HeuristicType.REDUCTION and (self.size_hints is not None) and (device_prop.major == 8):
                for triton_config, compiled_binary in zip(self.configs, compiled_binaries):
                    assert len(self.size_hints) == 2
                    xblock = triton_config.kwargs['XBLOCK']
                    rblock = triton_config.kwargs['RBLOCK']
                    total_block = (self.size_hints[0] + xblock - 1) // xblock
                    nreg = getattr(compiled_binary, 'n_regs', None)
                    if nreg is None:
                        continue
                    if rblock <= 64:
                        continue
                    if nreg <= 65536 // device_prop.max_threads_per_multi_processor:
                        continue
                    nreg_per_warp = nreg * 32
                    nreg_per_block = nreg_per_warp * triton_config.num_warps
                    max_blocks_per_sm = max(65536 // nreg_per_block, 1)
                    if total_block <= max_blocks_per_sm * device_prop.multi_processor_count:
                        continue
                    new_config = copy.deepcopy(triton_config)
                    new_config.kwargs['RBLOCK'] = rblock // 2
                    if new_config in seen_configs:
                        continue
                    seen_configs.add(new_config)
                    self.launchers.append(self._precompile_config(new_config, warm_cache_only_with_cc)[1])
            self.configs = None

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

    def bench(self, launcher, *args, grid, **kwargs):
        """Measure the performance of a given launcher"""
        if launcher.n_spills > config.triton.spill_threshold:
            log.debug('Skip config %s because of register spilling: %d', launcher.config, launcher.n_spills)
            return float('inf')
        stream = get_cuda_stream(torch.cuda.current_device())

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook({**dict(zip(self.arg_names, args)), **launcher.config.kwargs})
            cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
            launcher(*cloned_args, **cloned_kwargs, grid=grid, stream=stream)
        return do_bench(kernel_call, rep=40, fast_flush=True)

    def clone_args(self, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        from .compile_fx import clone_preserve_strides
        cloned_args = []
        for i, arg in enumerate(args):
            if self.fn.arg_names[i] in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_args.append(clone_preserve_strides(arg))
            else:
                cloned_args.append(arg)
        cloned_kwargs: Dict[str, Any] = {}
        for name, arg in kwargs.items():
            if name in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_kwargs[name] = clone_preserve_strides(arg)
            else:
                cloned_kwargs[name] = arg
        return (cloned_args, cloned_kwargs)

    @dynamo_timed
    def benchmark_all_configs(self, *args, **kwargs):
        timings = {launcher: self.bench(launcher, *args, **kwargs) for launcher in self.launchers}
        for k, v in timings.items():
            self.coordesc_tuner.cache_benchmark_result(k.config, v)
        if log.isEnabledFor(logging.DEBUG):
            log.debug('Benchmark all input configs get:')
            for k, v in timings.items():
                log.debug('%s: %f, nreg %d, nspill %d, #shared-mem %d', k.config, v, k.n_regs, k.n_spills, k.shared)
        return timings

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""
        timings = self.benchmark_all_configs(*args, **kwargs)
        self.launchers = [builtins.min(timings, key=timings.get)]
        if self.save_cache_hook:
            self.save_cache_hook(self.launchers[0].config)

    def save_cuda_kernel(self, grid, stream, launcher):
        if callable(grid):
            grid_x, grid_y, grid_z = grid(launcher.config.kwargs)
        else:
            grid_x, grid_y, grid_z = grid
        key = self.inductor_meta.get('kernel_name', None)
        assert key is not None, 'kernel_name can not be None'
        params = {'mangled_name': launcher.bin.metadata['name'], 'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z, 'x_block': launcher.config.kwargs.get('XBLOCK', 1), 'y_block': launcher.config.kwargs.get('YBLOCK', None), 'z_block': launcher.config.kwargs.get('ZBLOCK', None), 'num_warps': launcher.bin.num_warps, 'shared_mem': launcher.bin.shared, 'stream': stream, 'meta': launcher.config.kwargs}
        if torch.version.hip is None:
            CudaKernelParamCache.set(key, params, launcher.bin.asm['cubin'])
        else:
            import pathlib
            launcher.bin.asm['hsaco'] = pathlib.Path(launcher.bin.asm['hsaco_path']).read_bytes()
            CudaKernelParamCache.set(key, params, launcher.bin.asm['hsaco'])

    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """
        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate descnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        """
        if self.heuristic_type == HeuristicType.TEMPLATE or self.heuristic_type == HeuristicType.USER_AUTOTUNE:
            return launcher
        cloned_args, _ = self.clone_args(*args)
        config2launcher = {launcher.config: launcher}

        def benchmark_one_config(config):
            with self.lock:
                _, launcher = self._precompile_config(config, None)
            config2launcher[config] = launcher
            out = self.bench(launcher, *cloned_args, **kwargs)
            log.debug('COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d', launcher.config, out, launcher.n_regs, launcher.n_spills, launcher.shared)
            return out
        assert not (self.heuristic_type == HeuristicType.PERSISTENT_REDUCTION and 'RBLOCK' in launcher.config.kwargs), "Coordinate descent tuner relies on the assumption that persistent reduction's triton config does not have RBLOCK"
        best_config = self.coordesc_tuner.autotune(benchmark_one_config, launcher.config, None)
        best_config.found_by_coordesc = True
        if self.save_cache_hook:
            self.save_cache_hook(best_config, found_by_coordesc=True)
        return config2launcher.get(best_config)

    def run(self, *args, grid, stream, **kwargs):
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid, **kwargs)
        if not getattr(self.launchers[0].config, 'found_by_coordesc', False) and config.coordinate_descent_tuning:
            self.launchers = [self.coordinate_descent_tuning(self.launchers[0], *args, grid=grid, **kwargs)]
        launcher, = self.launchers
        if launcher.store_cubin:
            self.save_cuda_kernel(grid, stream, launcher)
        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook({**dict(zip(self.arg_names, args)), **launcher.config.kwargs, **kwargs})
        if autograd_profiler._is_profiler_enabled:
            with self.record_function_ctx:
                return launcher(*args, **kwargs, grid=grid, stream=stream)
        else:
            return launcher(*args, **kwargs, grid=grid, stream=stream)