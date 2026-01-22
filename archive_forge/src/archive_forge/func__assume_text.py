from __future__ import annotations
import logging # isort:skip
import json
from typing import (
from .exceptions import ValidationError
from .message import BufferHeader, Message
def _assume_text(self, fragment: Fragment) -> str:
    if not isinstance(fragment, str):
        raise ValidationError(f'expected text fragment but received binary fragment for {self._current_consumer.__name__}')
    return fragment