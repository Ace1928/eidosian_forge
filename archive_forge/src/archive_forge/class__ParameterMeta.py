import torch
from torch._C import _disabled_torch_function_impl
from collections import OrderedDict
class _ParameterMeta(torch._C._TensorMeta):

    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance) or (isinstance(instance, torch.Tensor) and getattr(instance, '_is_param', False))