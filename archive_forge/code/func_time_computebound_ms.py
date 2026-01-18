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
@property
def time_computebound_ms(self) -> float:
    assert self.time_ms > 0.0
    tflop = self.flop_count / 1000 ** 4
    if tflop == 0.0:
        return 0.0
    return min(self.time_ms, 1000 * tflop / self.hardware_tflops_limit)