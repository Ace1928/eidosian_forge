import torch
from torch.utils import _pytree as pytree
from typing import Optional
def _register_default_op(op, decorator):

    @decorator(op)
    def tensor_default_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the default tensor ops that
        behave the same as ``torch.Tensor`` such as ``torch.Tensor.shape`` or
        ``torch.Tensor.dtype``. We simply lower to the real op call with
        DisableTorchFunctionSubclass context like ``torch.Tensor.__torch_function__``
        to avoid recursions.
        """
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunctionSubclass():
            return op(*args, **kwargs)