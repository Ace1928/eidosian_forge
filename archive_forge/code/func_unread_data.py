import asyncio
import collections
import warnings
from typing import (
from .base_protocol import BaseProtocol
from .helpers import BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger
def unread_data(self, data: bytes) -> None:
    """rollback reading some data from stream, inserting it to buffer head."""
    warnings.warn('unread_data() is deprecated and will be removed in future releases (#3260)', DeprecationWarning, stacklevel=2)
    if not data:
        return
    if self._buffer_offset:
        self._buffer[0] = self._buffer[0][self._buffer_offset:]
        self._buffer_offset = 0
    self._size += len(data)
    self._cursor -= len(data)
    self._buffer.appendleft(data)
    self._eof_counter = 0