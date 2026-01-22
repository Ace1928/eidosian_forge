from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidUpgrade(InvalidHeader):
    """
    Raised when the Upgrade or Connection header isn't correct.

    """