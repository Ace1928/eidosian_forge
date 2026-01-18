import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import definitely_true, sym_eq
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from .schemas import MutationType
def was_tensor_updated(arg, new_arg):
    if is_traceable_wrapper_subclass(arg):
        assert is_traceable_wrapper_subclass(new_arg)
        attrs, _ = arg.__tensor_flatten__()
        new_attrs, _ = new_arg.__tensor_flatten__()
        assert attrs == new_attrs
        return any((was_tensor_updated(getattr(arg, attr), getattr(new_arg, attr)) for attr in attrs))
    else:
        return arg is not new_arg