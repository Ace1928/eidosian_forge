import sys
import typing
from collections.abc import Callable
from os import PathLike
from typing import (  # type: ignore
from typing_extensions import (
def update_model_forward_refs(model: Type[Any], fields: Iterable['ModelField'], json_encoders: Dict[Union[Type[Any], str, ForwardRef], AnyCallable], localns: 'DictStrAny', exc_to_suppress: Tuple[Type[BaseException], ...]=()) -> None:
    """
    Try to update model fields ForwardRefs based on model and localns.
    """
    if model.__module__ in sys.modules:
        globalns = sys.modules[model.__module__].__dict__.copy()
    else:
        globalns = {}
    globalns.setdefault(model.__name__, model)
    for f in fields:
        try:
            update_field_forward_refs(f, globalns=globalns, localns=localns)
        except exc_to_suppress:
            pass
    for key in set(json_encoders.keys()):
        if isinstance(key, str):
            fr: ForwardRef = ForwardRef(key)
        elif isinstance(key, ForwardRef):
            fr = key
        else:
            continue
        try:
            new_key = evaluate_forwardref(fr, globalns, localns or None)
        except exc_to_suppress:
            continue
        json_encoders[new_key] = json_encoders.pop(key)