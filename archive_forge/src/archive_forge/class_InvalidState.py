from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidState(WebSocketException, AssertionError):
    """
    Raised when an operation is forbidden in the current state.

    This exception is an implementation detail.

    It should never be raised in normal circumstances.

    """