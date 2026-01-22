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
def _broadcast_in_dim_meta(a: TensorLikeType, shape: ShapeType, broadcast_dimensions: Sequence[int]):
    assert isinstance(a, TensorLike)
    assert isinstance(shape, Sequence)
    assert isinstance(broadcast_dimensions, Sequence)
    assert a.ndim == len(broadcast_dimensions)
    assert len(shape) >= a.ndim

    def _greater_than_reduce(acc, x):
        assert isinstance(x, Dim)
        assert x > acc
        assert x < len(shape)
        return x
    reduce(_greater_than_reduce, broadcast_dimensions, -1)
    for idx, new_idx in enumerate(broadcast_dimensions):
        assert a.shape[idx] == 1 or a.shape[idx] == shape[new_idx]
    new_strides = []
    original_idx = 0
    for idx in range(len(shape)):
        if idx in broadcast_dimensions:
            if a.shape[original_idx] != shape[idx]:
                new_strides.append(0)
            else:
                new_strides.append(a.stride()[original_idx])
            original_idx = original_idx + 1
        elif shape[idx] != 1:
            new_strides.append(0)
        elif original_idx == a.ndim:
            new_strides.append(1)
        else:
            new_strides.append(a.stride()[original_idx] * a.size()[original_idx])
    return a.as_strided(shape, new_strides, a.storage_offset())