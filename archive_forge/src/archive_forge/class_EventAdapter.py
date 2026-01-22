from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass
from types import TracebackType
from sniffio import AsyncLibraryNotFoundError
from ..lowlevel import cancel_shielded_checkpoint, checkpoint, checkpoint_if_cancelled
from ._eventloop import get_async_backend
from ._exceptions import BusyResourceError, WouldBlock
from ._tasks import CancelScope
from ._testing import TaskInfo, get_current_task
class EventAdapter(Event):
    _internal_event: Event | None = None

    def __new__(cls) -> EventAdapter:
        return object.__new__(cls)

    @property
    def _event(self) -> Event:
        if self._internal_event is None:
            self._internal_event = get_async_backend().create_event()
        return self._internal_event

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._internal_event is not None and self._internal_event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def statistics(self) -> EventStatistics:
        if self._internal_event is None:
            return EventStatistics(tasks_waiting=0)
        return self._internal_event.statistics()