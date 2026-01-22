from __future__ import annotations
import http
from typing import Optional
from . import datastructures, frames, http11
from .typing import StatusLike
class InvalidHeaderFormat(InvalidHeader):
    """
    Raised when an HTTP header cannot be parsed.

    The format of the header doesn't match the grammar for that header.

    """

    def __init__(self, name: str, error: str, header: str, pos: int) -> None:
        super().__init__(name, f'{error} at {pos} in {header}')