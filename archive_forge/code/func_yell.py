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
def yell():
    raise RuntimeError(f'impl_backward(output_differentiability): expected output_differentiability to be a list of bools with length equal to the number of outputs of this CustomOp got: {output_differentiability}')