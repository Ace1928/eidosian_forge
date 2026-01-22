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
class GemmOpComputeFlopsBmm(GemmOpComputeFlops):

    def _get_mnk(self, inputs: List[Any]) -> Tuple[int, int, int]:
        a, b = (inputs[0], inputs[1])
        assert a.ndim == 3
        assert b.ndim == 3
        bs = max(inputs[0].shape[0], inputs[1].shape[0])
        return (bs * a.shape[1], b.shape[-1], b.shape[-2])