import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def _outer_to_inner_dim(ndim, dim):
    assert dim >= 0 and dim < ndim
    return 0 if dim < 2 else dim - 1