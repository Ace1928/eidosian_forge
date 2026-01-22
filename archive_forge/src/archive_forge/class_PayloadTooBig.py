from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class PayloadTooBig(WebSocketException):
    """
    Raised when receiving a frame with a payload exceeding the maximum size.

    """