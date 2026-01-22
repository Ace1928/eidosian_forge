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
def _threshold(input: Tensor, threshold: float, value: float, inplace: bool=False) -> Tensor:
    """Apply a threshold to each element of the input Tensor.

    See :class:`~torch.nn.Threshold` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(_threshold, (input,), input, threshold, value, inplace=inplace)
    if inplace:
        result = _VF.threshold_(input, threshold, value)
    else:
        result = _VF.threshold(input, threshold, value)
    return result