from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar
import attrs
from .. import _core
from .._deprecate import deprecated
from .._util import final
def _get_batch_protected(self) -> list[T]:
    data = self._data.copy()
    self._data.clear()
    self._can_get = False
    return data