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
@staticmethod
def _get_catch_all_field(cls) -> Field:
    cls_globals = vars(sys.modules[cls.__module__])
    types = get_type_hints(cls, globalns=cls_globals)
    catch_all_fields = list(filter(lambda f: types[f.name] == Optional[CatchAllVar], fields(cls)))
    number_of_catch_all_fields = len(catch_all_fields)
    if number_of_catch_all_fields == 0:
        raise UndefinedParameterError('No field of type dataclasses_json.CatchAll defined')
    elif number_of_catch_all_fields > 1:
        raise UndefinedParameterError(f'Multiple catch-all fields supplied: {number_of_catch_all_fields}.')
    else:
        return catch_all_fields[0]