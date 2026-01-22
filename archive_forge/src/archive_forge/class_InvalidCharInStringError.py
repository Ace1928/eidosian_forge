from __future__ import annotations
from typing import Collection
class InvalidCharInStringError(ParseError):
    """
    The string being parsed contains an invalid character.
    """

    def __init__(self, line: int, col: int, char: str) -> None:
        message = f'Invalid character {repr(char)} in string'
        super().__init__(line, col, message=message)