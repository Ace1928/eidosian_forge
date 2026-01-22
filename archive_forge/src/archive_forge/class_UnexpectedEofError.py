from __future__ import annotations
from typing import Collection
class UnexpectedEofError(ParseError):
    """
    The TOML being parsed ended before the end of a statement.
    """

    def __init__(self, line: int, col: int) -> None:
        message = 'Unexpected end of file'
        super().__init__(line, col, message=message)