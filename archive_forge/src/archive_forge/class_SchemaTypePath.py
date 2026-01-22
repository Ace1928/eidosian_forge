from __future__ import annotations
from typing import Any, Callable, NamedTuple
from pydantic_core import CoreConfig, CoreSchema, ValidationError
from typing_extensions import Literal, Protocol, TypeAlias
class SchemaTypePath(NamedTuple):
    """Path defining where `schema_type` was defined, or where `TypeAdapter` was called."""
    module: str
    name: str