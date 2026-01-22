from __future__ import annotations
import typing as t
from datetime import datetime
class BadData(Exception):
    """Raised if bad data of any sort was encountered. This is the base
    for all exceptions that ItsDangerous defines.

    .. versionadded:: 0.15
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message