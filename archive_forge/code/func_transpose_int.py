import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.transpose.int, 'self: jt, dim0: any, dim1: any')
def transpose_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    from torch._prims_common import canonicalize_dims
    inp = new_kwargs.pop('input')
    dim0, dim1 = canonicalize_dims(inp.dim(), (new_kwargs['dim0'], new_kwargs['dim1']))
    if dim0 == inp._ragged_idx or dim1 == inp._ragged_idx:
        if dim0 == 0 or dim1 == 0:
            raise ValueError('Transpose is not supported on the batch dimension for jagged NT')
        if dim0 == inp._ragged_idx:
            to_dim = dim1
        else:
            to_dim = dim0
        return NestedTensor(inp.values().transpose(_outer_to_inner_dim(len(inp._size), dim0), _outer_to_inner_dim(len(inp._size), dim1)), **extract_kwargs(inp), _ragged_idx=to_dim)
    new_kwargs['dim0'] = _wrap_jagged_dim(inp.dim(), new_kwargs['dim0'], 'transpose')
    new_kwargs['dim1'] = _wrap_jagged_dim(inp.dim(), new_kwargs['dim1'], 'transpose')
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))