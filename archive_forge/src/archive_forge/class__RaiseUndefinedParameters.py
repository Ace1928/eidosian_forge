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
class _RaiseUndefinedParameters(_UndefinedParameterAction):
    """
    This action raises UndefinedParameterError if it encounters an undefined
    parameter during initialization.
    """

    @staticmethod
    def handle_from_dict(cls, kvs: Dict) -> Dict[str, Any]:
        known, unknown = _UndefinedParameterAction._separate_defined_undefined_kvs(cls=cls, kvs=kvs)
        if len(unknown) > 0:
            raise UndefinedParameterError(f'Received undefined initialization arguments {unknown}')
        return known