import dataclasses
import functools
import inspect
import sys
import typing
import weakref
from torchgen.model import FunctionSchema, OperatorName, SchemaKind, BaseType, ListType, BaseTy
import torch
import torch._C as _C
import torch.library as library
from torch._library.abstract_impl import AbstractImplCtx
from torch.library import get_ctx
from .autograd import autograd_kernel_indirection, construct_autograd_kernel
def _find_custom_op(qualname, also_check_torch_library=False):
    if qualname in global_registry:
        return global_registry[qualname]
    if not also_check_torch_library:
        raise RuntimeError(f'Could not find custom op "{qualname}". Did you register it via the torch._custom_ops API?')
    overload = get_op(qualname)
    result = custom_op_from_existing(overload)
    return result