from __future__ import annotations
from typing import Collection
class InvalidDateTimeError(ParseError):
    """
    A datetime field was improperly specified.
    """

    def __init__(self, line: int, col: int) -> None:
        message = 'Invalid datetime'
        super().__init__(line, col, message=message)