from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidParameterName(NegotiationError):
    """
    Raised when a parameter name in an extension header is invalid.

    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f'invalid parameter name: {self.name}'