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
def validate_device_type(device_type: str) -> None:
    if device_type not in SUPPORTED_DEVICE_TYPE_TO_KEY:
        raise ValueError(f'CustomOp.impl(device_types=[{device_type}, ...]): we only support device_type in {SUPPORTED_DEVICE_TYPE_TO_KEY.keys()}.')