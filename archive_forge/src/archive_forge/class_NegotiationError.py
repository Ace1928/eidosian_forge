from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class NegotiationError(InvalidHandshake):
    """
    Raised when negotiating an extension fails.

    """