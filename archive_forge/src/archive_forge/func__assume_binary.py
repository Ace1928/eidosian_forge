from __future__ import annotations
import logging # isort:skip
import json
from typing import (
from .exceptions import ValidationError
from .message import BufferHeader, Message
def _assume_binary(self, fragment: Fragment) -> bytes:
    if not isinstance(fragment, bytes):
        raise ValidationError(f'expected binary fragment but received text fragment for {self._current_consumer.__name__}')
    return fragment