from __future__ import annotations
import sys as _sys
from typing import Any as _Any
from ._pydantic_core import (
from .core_schema import CoreConfig, CoreSchema, CoreSchemaType, ErrorType
class MultiHostHost(_TypedDict):
    """
    A host part of a multi-host URL.
    """
    username: str | None
    'The username part of this host, or `None`.'
    password: str | None
    'The password part of this host, or `None`.'
    host: str | None
    'The host part of this host, or `None`.'
    port: int | None
    'The port part of this host, or `None`.'