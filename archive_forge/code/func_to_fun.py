import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def to_fun(t):
    if isinstance(t, Tensor):
        if is_traceable_wrapper_subclass(t):
            out = transform_subclass(t, lambda _, inner_t: to_fun(inner_t))
            torch._mirror_autograd_meta_to(t, out)
            return out
        else:
            return FunctionalTensor.to_functional(t)
    else:
        return t