from __future__ import annotations
from typing import Collection
class MixedArrayTypesError(ParseError):
    """
    An array was found that had two or more element types.
    """

    def __init__(self, line: int, col: int) -> None:
        message = 'Mixed types found in array'
        super().__init__(line, col, message=message)