import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Set, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.profiler
import torch.utils.hooks
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.utils._pytree import tree_map
from ..ops.common import FUNC_TO_XFORMERS_OPERATOR
from .device_limits import get_device_limits
from .profiler import _Profiler
def _hardware_tflops_membw_limit(self, args: Tuple[Any, ...], outputs: Tuple[Any, ...]) -> Tuple[float, float]:
    device = None
    dtypes: List[torch.dtype] = []
    for a in itertools.chain(outputs, args):
        if isinstance(a, torch.Tensor):
            if device is None:
                device = a.device
            dtypes.append(a.dtype)
    limits = get_device_limits(device)
    dtypes = [dt for dt in dtypes if dt in limits.gemm_tflops]
    if not dtypes or device is None:
        return (math.inf, math.inf)
    dtype = dtypes[0]
    if torch.is_autocast_enabled() and dtype is torch.float32:
        dtype = torch.get_autocast_gpu_dtype()
    return (limits.gemm_tflops[dtype], limits.gmem_bandwidth)