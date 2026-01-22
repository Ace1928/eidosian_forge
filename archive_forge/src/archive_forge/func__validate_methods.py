import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor
def _validate_methods(self):
    forbidden_overrides = ['size', 'stride', 'dtype', 'layout', 'device', 'requires_grad']
    for el in forbidden_overrides:
        if getattr(self.__class__, el) is not getattr(torch.Tensor, el):
            raise RuntimeError(f'Subclass {self.__class__.__name__} is overwriting the property {el} but this is not allowed as such change would not be reflected to c++ callers.')