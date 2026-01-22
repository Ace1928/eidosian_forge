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
def _register_autograd_kernel_indirection(self):
    assert not self._registered_autograd_kernel_indirection
    self._lib.impl(self._opname, autograd_kernel_indirection(weakref.proxy(self)), 'Autograd')
    self._registered_autograd_kernel_indirection = True