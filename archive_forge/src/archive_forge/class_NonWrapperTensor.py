import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor
class NonWrapperTensor(torch.Tensor):

    def __new__(cls, data):
        t = torch.Tensor._make_subclass(cls, data)
        t.extra_state = {'last_func_called': None}
        return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        result = super().__torch_function__(func, types, args, kwargs)
        if isinstance(result, cls):
            if func is torch.Tensor.__deepcopy__:
                result.extra_state = deepcopy(args[0].extra_state)
            else:
                result.extra_state = {'last_func_called': func.__name__}
        return result

    def new_empty(self, shape):
        return type(self)(torch.empty(shape))