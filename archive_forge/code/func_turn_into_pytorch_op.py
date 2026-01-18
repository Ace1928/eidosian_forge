import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
def turn_into_pytorch_op(fn: ClsT, dispatch_key: str) -> ClsT:
    from .. import get_python_lib

    def render_arg_type(annotation) -> str:
        if get_origin(annotation) is Union:
            inner_types = [t for t in get_args(annotation) if t is not type(None)]
            if len(inner_types) == 1:
                return f'{render_arg_type(inner_types[0])}?'
        if get_origin(annotation) is list:
            inner_type, = get_args(annotation)
            return f'{render_arg_type(inner_type)}[]'
        if get_origin(annotation) is tuple:
            return '(' + ', '.join([render_arg_type(t) for t in get_args(annotation)]) + ')'
        if get_origin(annotation) is Annotated:
            inner_type, annotation = get_args(annotation)
            if isinstance(annotation, Alias):
                alias = annotation.name + ('!' if annotation.write else '')
                return f'{render_arg_type(inner_type)}({alias})'
        if annotation is torch.Tensor:
            return 'Tensor'
        if annotation is bool:
            return 'bool'
        if annotation is int:
            return 'int'
        if annotation is float:
            return 'float'
        if annotation is torch.dtype:
            return 'ScalarType'
        if annotation is torch.distributed.ProcessGroup:
            return '__torch__.torch.classes.c10d.ProcessGroup'
        assert False, f'Unable to parse annotation: `{annotation}`'

    def render_default_value(default):
        if default is inspect.Parameter.empty:
            return ''
        return f' = {default!r}'
    sign = inspect.signature(fn)
    arguments = [f'{render_arg_type(arg.annotation)} {arg.name}{render_default_value(arg.default)}' for arg in sign.parameters.values()]
    op_name = fn.__name__
    definition = f'{op_name}({', '.join(arguments)}) -> {render_arg_type(sign.return_annotation)}'

    def callee(*args, **kwargs):
        ba = sign.bind(*args, **kwargs)
        for name, value in ba.arguments.items():
            if sign.parameters[name].annotation is torch.distributed.ProcessGroup:
                from .._C import unbox_process_group
                ba.arguments[name] = unbox_process_group(value)
        return fn(*ba.args, **ba.kwargs)
    xformers_lib = get_python_lib()
    xformers_lib.define(definition)
    xformers_lib.impl(op_name, callee, dispatch_key)
    dispatcher_impl = getattr(getattr(torch.ops, xformers_lib.ns), op_name)

    @wraps(fn)
    def caller(*args, **kwargs):
        ba = sign.bind(*args, **kwargs)
        for name, value in ba.arguments.items():
            if sign.parameters[name].annotation is torch.distributed.ProcessGroup:
                from .._C import box_process_group
                ba.arguments[name] = box_process_group(value)
        return dispatcher_impl(*ba.args, **ba.kwargs)
    return caller