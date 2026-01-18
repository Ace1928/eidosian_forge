import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.unsqueeze.default, 'self: jt, dim: any')
def unsqueeze_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    values = inp._values
    offsets = inp.offsets
    dim = new_kwargs['dim']
    new_kwargs['dim'] = _wrap_jagged_dim(len(inp.shape) + 1, dim, 'unsqueeze')
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))