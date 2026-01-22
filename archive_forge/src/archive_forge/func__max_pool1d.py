from typing import Callable, List, Optional, Tuple, Union
import math
import warnings
import importlib
import torch
from torch import _VF
from torch import sym_int as _sym_int
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes, sparse_support_notes
from typing import TYPE_CHECKING
from .._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
from ..overrides import (
from . import _reduction as _Reduction
from . import grad  # noqa: F401
from .modules import utils
from .modules.utils import _single, _pair, _triple, _list_with_default
def _max_pool1d(input: Tensor, kernel_size: BroadcastingList1[int], stride: Optional[BroadcastingList1[int]]=None, padding: BroadcastingList1[int]=0, dilation: BroadcastingList1[int]=1, ceil_mode: bool=False, return_indices: bool=False) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(max_pool1d, (input,), input, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)