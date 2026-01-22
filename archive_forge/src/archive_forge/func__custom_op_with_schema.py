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
def _custom_op_with_schema(qualname, schema):
    ns, name = qualname.split('::')
    schema_str = f'{name}{schema}'
    function_schema = FunctionSchema.parse(schema_str)
    validate_schema(function_schema)
    lib = library.Library(ns, 'FRAGMENT')
    lib.define(schema_str)
    ophandle = find_ophandle_or_throw(ns, function_schema.name)
    result = CustomOp(lib, ns, function_schema, name, ophandle, _private_access=True)
    result._register_autograd_kernel_indirection()
    torch._C._dispatch_set_report_error_callback(ophandle, functools.partial(report_error_callback, weakref.proxy(result)))
    return get_op(qualname)