import contextlib
import itertools
import operator
import weakref
from enum import Enum
from functools import partial, reduce
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch
import torch._prims_common as utils
import torch.library
from torch import sym_float, Tensor, TypedStorage
from torch._C import _get_default_device
from torch._prims.debug_prims import register_debug_prims
from torch._prims.rng_prims import register_rng_prims
from torch._prims_common import (
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
def _collapse_view_helper(a: TensorLikeType, start: int, end: int) -> Tuple[Optional[ShapeType], Optional[StrideType]]:
    assert isinstance(a, TensorLike)
    _validate_collapse_args(a, start, end)
    if a.ndim == 0:
        shape = (1,)
        strides = (1,)
    else:
        shape = a.shape
        strides = a.stride()
    if a.ndim == 0 or end == start:
        return (shape, strides)
    length = shape[end]
    stride = strides[end]
    for idx in range(end - 1, start - 1, -1):
        if shape[idx] == 0 or shape[idx + 1] == 0:
            length = 0
            stride = 0
            break
        if shape[idx] == 1:
            continue
        length = length * shape[idx]
        stride = min(stride, strides[idx])
        if a.numel() > 0 and shape[idx + 1] != 1 and (not strides[idx] == strides[idx + 1] * shape[idx + 1]):
            return (None, None)
    new_shape = shape[:start] + (length,) + shape[end + 1:]
    new_strides = strides[:start] + (stride,) + strides[end + 1:]
    if a.numel() == 0:
        new_strides = utils.make_contiguous_strides_for(new_shape)
    return (new_shape, new_strides)