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
@property
def _limiter(self) -> CapacityLimiter:
    if self._internal_limiter is None:
        self._internal_limiter = get_async_backend().create_capacity_limiter(self._total_tokens)
    return self._internal_limiter