import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def _wrap_jagged_dim(ndim, dim, op_name):
    from torch._prims_common import canonicalize_dims
    wrapped = canonicalize_dims(ndim, dim)
    if wrapped < 2:
        raise RuntimeError(f'{op_name}(): not supported for NestedTensor on dim=0 or dim=1')
    return _outer_to_inner_dim(ndim, wrapped)