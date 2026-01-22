from dataclasses import Field, MISSING, _FIELDS, _FIELD, _FIELD_INITVAR  # type: ignore
from typing import Type, Any, TypeVar, List
from .data import Data
from .types import is_optional
class DefaultValueNotFoundError(Exception):
    pass