import abc
import dataclasses
import functools
import inspect
import sys
from dataclasses import Field, fields
from typing import Any, Callable, Dict, Optional, Tuple, Union, Type, get_type_hints
from enum import Enum
from marshmallow.exceptions import ValidationError  # type: ignore
from dataclasses_json.utils import CatchAllVar
@functools.wraps(obj.__init__)
def _catch_all_init(self, *args, **kwargs):
    known_kwargs, unknown_kwargs = _CatchAllUndefinedParameters._separate_defined_undefined_kvs(obj, kwargs)
    num_params_takeable = len(init_signature.parameters) - 1
    if _CatchAllUndefinedParameters._get_catch_all_field(obj).name not in known_kwargs:
        num_params_takeable -= 1
    num_args_takeable = num_params_takeable - len(known_kwargs)
    args, unknown_args = (args[:num_args_takeable], args[num_args_takeable:])
    bound_parameters = init_signature.bind_partial(self, *args, **known_kwargs)
    unknown_args = {f'_UNKNOWN{i}': v for i, v in enumerate(unknown_args)}
    arguments = bound_parameters.arguments
    arguments.update(unknown_args)
    arguments.update(unknown_kwargs)
    arguments.pop('self', None)
    final_parameters = _CatchAllUndefinedParameters.handle_from_dict(obj, arguments)
    original_init(self, **final_parameters)