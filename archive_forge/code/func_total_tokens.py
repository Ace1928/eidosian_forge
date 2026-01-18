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
@total_tokens.setter
def total_tokens(self, value: float) -> None:
    if not isinstance(value, int) and value is not math.inf:
        raise TypeError('total_tokens must be an int or math.inf')
    elif value < 1:
        raise ValueError('total_tokens must be >= 1')
    if self._internal_limiter is None:
        self._total_tokens = value
        return
    self._limiter.total_tokens = value