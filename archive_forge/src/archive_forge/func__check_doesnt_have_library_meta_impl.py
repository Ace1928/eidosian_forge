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
def _check_doesnt_have_library_meta_impl(self):
    if self._has_impl('abstract'):
        return
    if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, 'CompositeExplicitAutograd') and (not _C._dispatch_has_kernel_for_dispatch_key(self._qualname, 'Meta')):
        return
    if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, 'CompositeImplicitAutograd'):
        raise RuntimeError(f'impl_abstract(...): the operator {self._qualname} already has an implementation for this device type via a pre-existing registration to DispatchKey::CompositeImplicitAutograd.CompositeImplicitAutograd operators do not need an abstract impl; instead, the operator will decompose into its constituents and those can have abstract impls defined on them.')
    if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, 'Meta'):
        raise RuntimeError(f"impl_abstract(...): the operator {self._qualname} already has an DispatchKey::Meta implementation via a pre-existing torch.library or TORCH_LIBRARY registration. Please either remove that registration or don't call impl_abstract.")