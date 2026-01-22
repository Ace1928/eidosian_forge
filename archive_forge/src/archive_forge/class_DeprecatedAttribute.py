from __future__ import annotations
import sys
import warnings
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, TypeVar
import attrs
@attrs.frozen(slots=False)
class DeprecatedAttribute:
    _not_set: ClassVar[object] = object()
    value: object
    version: str
    issue: int | None
    instead: object = _not_set