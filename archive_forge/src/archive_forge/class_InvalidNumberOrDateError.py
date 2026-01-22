from __future__ import annotations
from typing import Collection
class InvalidNumberOrDateError(ParseError):
    """
    A numeric or date field was improperly specified.
    """

    def __init__(self, line: int, col: int) -> None:
        message = 'Invalid number or date format'
        super().__init__(line, col, message=message)