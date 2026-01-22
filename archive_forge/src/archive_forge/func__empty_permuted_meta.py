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
def _empty_permuted_meta(shape: ShapeType, physical_layout: DimsSequenceType, *, dtype: torch.dtype, device: torch.device, requires_grad: bool) -> TensorLikeType:
    p_strides = utils.make_contiguous_strides_for([shape[l] for l in physical_layout])
    dim = len(shape)
    torch._check(len(physical_layout) == dim, lambda: f'Number of dimensions in the tensor input does not match the length of the physical layout; i.e. len(size) = {dim} is not equal to len(physical_layout) = {len(physical_layout)}')
    strides = [0] * len(shape)
    seen_dims = set()
    for p, l in enumerate(physical_layout):
        torch._check(0 <= l < dim, lambda: f'Dimension out of range (expected to be between 0 and {dim - 1}, but got {l} at index {p}).  NB: negative dims not currently supported; file an issue if you want it.')
        torch._check(l not in seen_dims, lambda: 'Duplicate dim not allowed')
        strides[l] = p_strides[p]
        seen_dims.add(l)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)