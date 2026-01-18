import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.where.self, 'condition: jt, self: jt, other: jt')
def where_self(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    condition = new_kwargs.pop('condition')
    inp = new_kwargs.pop('input')
    other = new_kwargs.pop('other')
    assert condition.shape == other.shape == inp.shape
    return NestedTensor(func(condition._values, inp._values, other._values, **new_kwargs), **extract_kwargs(condition))