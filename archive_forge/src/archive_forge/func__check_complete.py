from __future__ import annotations
import logging # isort:skip
import json
from typing import (
from .exceptions import ValidationError
from .message import BufferHeader, Message
def _check_complete(self) -> None:
    if self._partial and self._partial.complete:
        self._message = self._partial
        self._current_consumer = self._HEADER
    else:
        self._current_consumer = self._BUFFER_HEADER