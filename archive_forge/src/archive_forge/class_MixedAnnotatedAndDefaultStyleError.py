import inspect
import sys
from copy import copy
from typing import Any, Callable, Dict, List, Tuple, Type, cast, get_type_hints
from typing_extensions import Annotated
from ._typing import get_args, get_origin
from .models import ArgumentInfo, OptionInfo, ParameterInfo, ParamMeta
class MixedAnnotatedAndDefaultStyleError(Exception):
    argument_name: str
    annotated_param_type: Type[ParameterInfo]
    default_param_type: Type[ParameterInfo]

    def __init__(self, argument_name: str, annotated_param_type: Type[ParameterInfo], default_param_type: Type[ParameterInfo]):
        self.argument_name = argument_name
        self.annotated_param_type = annotated_param_type
        self.default_param_type = default_param_type

    def __str__(self) -> str:
        annotated_param_type_str = _param_type_to_user_string(self.annotated_param_type)
        default_param_type_str = _param_type_to_user_string(self.default_param_type)
        msg = f'Cannot specify {annotated_param_type_str} in `Annotated` and'
        if self.annotated_param_type is self.default_param_type:
            msg += ' default value'
        else:
            msg += f' {default_param_type_str} as a default value'
        msg += f' together for {self.argument_name!r}'
        return msg