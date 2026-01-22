import warnings
import torch
from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor
def _torch_grad_reduce(fn):

    def grad_reduce(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return _torch_reduce_all(fn)(args[0])
        input, dim, keepdim, dtype = _reduce_dim_args(*args, **kwargs)
        return _torch_reduce_dim(fn)(input, dim, keepdim, dtype)
    return grad_reduce