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
@dataclass
class _OpInfoAggregated:
    is_exact_flop: bool = True
    total_flop_count: float = 0.0
    total_io_bytes: int = 0
    total_time_ms: float = 0.0
    total_time_membound_ms: float = 0.0
    total_time_computebound_ms: float = 0.0
    num: int = 0
    stacktraces: List[Tuple[str, ...]] = field(default_factory=list)

    def add(self, op: _OpInfo) -> None:
        self.total_flop_count += op.flop_count
        self.total_time_ms += op.time_ms
        self.total_io_bytes += op.io_bytes
        self.total_time_membound_ms += op.time_membound_ms
        self.total_time_computebound_ms += op.time_computebound_ms
        self.num += 1
        self.is_exact_flop = op.is_exact_flop
        self.stacktraces.append(op.stacktrace)

    def as_dict(self, **kwargs) -> Dict[str, Any]:
        mem_bound = min(1, self.total_time_membound_ms / self.total_time_ms)
        tflops = self.total_flop_count / (self.total_time_ms / 1000) / 1000 ** 4
        compute_bound = min(1, self.total_time_computebound_ms / self.total_time_ms)
        return {'is_exact_flop': self.is_exact_flop, 'total_flop_count': self.total_flop_count, 'total_time_ms': self.total_time_ms, 'total_io_bytes': self.total_io_bytes, 'num': self.num, 'Tflops': tflops, 'mem_bound': mem_bound, 'compute_bound': compute_bound, **kwargs}