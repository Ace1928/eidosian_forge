from __future__ import annotations
import builtins
import time
from typing import Dict
from ..testing import do_bench
from .jit import KernelInterface
def _bench(self, *args, config, **meta):
    conflicts = meta.keys() & config.kwargs.keys()
    if conflicts:
        raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}. Make sure that you don't re-define auto-tuned symbols.")
    current = dict(meta, **config.kwargs)
    full_nargs = {**self.nargs, **current}

    def kernel_call():
        if config.pre_hook:
            config.pre_hook(full_nargs)
        self.pre_hook(args)
        self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, num_ctas=config.num_ctas, enable_warp_specialization=config.enable_warp_specialization, **current)
        self.post_hook(args)
    try:
        return do_bench(kernel_call, warmup=self.warmup, rep=self.rep, quantiles=(0.5, 0.2, 0.8))
    except OutOfResources:
        return [float('inf'), float('inf'), float('inf')]