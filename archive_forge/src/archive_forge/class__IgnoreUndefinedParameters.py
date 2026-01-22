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
class _IgnoreUndefinedParameters(_UndefinedParameterAction):
    """
    This action does nothing when it encounters undefined parameters.
    The undefined parameters can not be retrieved after the class has been
    created.
    """

    @staticmethod
    def handle_from_dict(cls, kvs: Dict) -> Dict[str, Any]:
        known_given_parameters, _ = _UndefinedParameterAction._separate_defined_undefined_kvs(cls=cls, kvs=kvs)
        return known_given_parameters

    @staticmethod
    def create_init(obj) -> Callable:
        original_init = obj.__init__
        init_signature = inspect.signature(original_init)

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
        return _ignore_init