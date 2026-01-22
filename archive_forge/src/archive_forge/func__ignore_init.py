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
def _ignore_init(self, *args, **kwargs):
    known_kwargs, _ = _CatchAllUndefinedParameters._separate_defined_undefined_kvs(obj, kwargs)
    num_params_takeable = len(init_signature.parameters) - 1
    num_args_takeable = num_params_takeable - len(known_kwargs)
    args = args[:num_args_takeable]
    bound_parameters = init_signature.bind_partial(self, *args, **known_kwargs)
    bound_parameters.apply_defaults()
    arguments = bound_parameters.arguments
    arguments.pop('self', None)
    final_parameters = _IgnoreUndefinedParameters.handle_from_dict(obj, arguments)
    original_init(self, **final_parameters)