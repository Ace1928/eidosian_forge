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
class DispatcherWithoutBrokenFuncs(TorchDispatchMode):
    TENSOR_FUNCS_NO_DISPATCH = ['record_stream']

    def __enter__(self) -> None:
        self._pt_impls = {}
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        return super().__exit__(exc_type, exc_val, exc_tb)