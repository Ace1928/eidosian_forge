from __future__ import annotations
import dataclasses
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import PydanticUndefined
from ._config import ConfigWrapper
from ._utils import is_valid_identifier
def _process_param_defaults(param: Parameter) -> Parameter:
    """Modify the signature for a parameter in a dataclass where the default value is a FieldInfo instance.

    Args:
        param (Parameter): The parameter

    Returns:
        Parameter: The custom processed parameter
    """
    from ..fields import FieldInfo
    param_default = param.default
    if isinstance(param_default, FieldInfo):
        annotation = param.annotation
        if annotation == 'Any':
            annotation = Any
        default = param_default.default
        if default is PydanticUndefined:
            if param_default.default_factory is PydanticUndefined:
                default = Signature.empty
            else:
                default = dataclasses._HAS_DEFAULT_FACTORY
        return param.replace(annotation=annotation, name=_field_name_for_signature(param.name, param_default), default=default)
    return param