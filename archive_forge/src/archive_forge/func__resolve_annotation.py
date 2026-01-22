from inspect import isclass, signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.decorators import undoc
def _resolve_annotation(annotation, sig: Signature, func: Callable, node: ast.Call, context: EvaluationContext):
    """Resolve annotation created by user with `typing` module and custom objects."""
    annotation = _eval_node_name(annotation, context) if isinstance(annotation, str) else annotation
    origin = get_origin(annotation)
    if annotation is Self and hasattr(func, '__self__'):
        return func.__self__
    elif origin is Literal:
        type_args = get_args(annotation)
        if len(type_args) == 1:
            return type_args[0]
    elif annotation is LiteralString:
        return ''
    elif annotation is AnyStr:
        index = None
        for i, (key, value) in enumerate(sig.parameters.items()):
            if value.annotation is AnyStr:
                index = i
                break
        if index is not None and index < len(node.args):
            return eval_node(node.args[index], context)
    elif origin is TypeGuard:
        return bool()
    elif origin is Union:
        attributes = [attr for type_arg in get_args(annotation) for attr in dir(_resolve_annotation(type_arg, sig, func, node, context))]
        return _Duck(attributes=dict.fromkeys(attributes))
    elif is_typeddict(annotation):
        return _Duck(attributes=dict.fromkeys(dir(dict())), items={k: _resolve_annotation(v, sig, func, node, context) for k, v in annotation.__annotations__.items()})
    elif hasattr(annotation, '_is_protocol'):
        return _Duck(attributes=dict.fromkeys(dir(annotation)))
    elif origin is Annotated:
        type_arg = get_args(annotation)[0]
        return _resolve_annotation(type_arg, sig, func, node, context)
    elif isinstance(annotation, NewType):
        return _eval_or_create_duck(annotation.__supertype__, node, context)
    elif isinstance(annotation, TypeAliasType):
        return _eval_or_create_duck(annotation.__value__, node, context)
    else:
        return _eval_or_create_duck(annotation, node, context)