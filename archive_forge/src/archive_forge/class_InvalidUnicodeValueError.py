from __future__ import annotations
from typing import Collection
class InvalidUnicodeValueError(ParseError):
    """
    A unicode code was improperly specified.
    """

    def __init__(self, line: int, col: int) -> None:
        message = 'Invalid unicode value'
        super().__init__(line, col, message=message)