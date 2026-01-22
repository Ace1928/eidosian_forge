from __future__ import annotations
import sys as _sys
from typing import Any as _Any
from ._pydantic_core import (
from .core_schema import CoreConfig, CoreSchema, CoreSchemaType, ErrorType
class InitErrorDetails(_TypedDict):
    type: str | PydanticCustomError
    'The type of error that occurred, this should a "slug" identifier that changes rarely or never.'
    loc: _NotRequired[tuple[int | str, ...]]
    'Tuple of strings and ints identifying where in the schema the error occurred.'
    input: _Any
    'The input data at this `loc` that caused the error.'
    ctx: _NotRequired[dict[str, _Any]]
    '\n    Values which are required to render the error message, and could hence be useful in rendering custom error messages.\n    Also useful for passing custom error data forward.\n    '