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
def _destroy(self):
    del self._lib
    opnamespace = getattr(torch.ops, self._cpp_ns)
    if hasattr(opnamespace, self._opname):
        delattr(opnamespace, self._opname)
    del global_registry[self._qualname]