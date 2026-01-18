import torch
from .core import _map_mt_args_kwargs, _wrap_result
def unary_fn(*args, **kwargs):
    return _unary_helper(fn, args, kwargs, inplace=True)