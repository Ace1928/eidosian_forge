from __future__ import annotations
import math
from typing import TYPE_CHECKING, Protocol
import attrs
import trio
from . import _core
from ._core import Abort, ParkingLot, RaiseCancelT, enable_ki_protection
from ._util import final
def _wake_waiters(self) -> None:
    available = self._total_tokens - len(self._borrowers)
    for woken in self._lot.unpark(count=available):
        self._borrowers.add(self._pending_borrowers.pop(woken))