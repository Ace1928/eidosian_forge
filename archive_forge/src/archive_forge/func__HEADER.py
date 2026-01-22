from __future__ import annotations
import logging # isort:skip
import json
from typing import (
from .exceptions import ValidationError
from .message import BufferHeader, Message
def _HEADER(self, fragment: Fragment) -> None:
    self._message = None
    self._partial = None
    self._fragments = [self._assume_text(fragment)]
    self._current_consumer = self._METADATA