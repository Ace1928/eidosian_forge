from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, overload
from weakref import WeakKeyDictionary
from ._core._eventloop import get_async_backend
@property
def _current_vars(self) -> dict[str, T]:
    token = current_token()
    try:
        return _run_vars[token]
    except KeyError:
        run_vars = _run_vars[token] = {}
        return run_vars