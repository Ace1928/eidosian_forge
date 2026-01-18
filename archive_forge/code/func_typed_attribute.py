from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, final, overload
from ._exceptions import TypedAttributeLookupError
def typed_attribute() -> Any:
    """Return a unique object, used to mark typed attributes."""
    return object()