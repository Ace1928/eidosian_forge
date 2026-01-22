from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
def _check_max_bytes(self, max_bytes: int | None) -> None:
    if max_bytes is None:
        return
    max_bytes = operator.index(max_bytes)
    if max_bytes < 1:
        raise ValueError('max_bytes must be >= 1')