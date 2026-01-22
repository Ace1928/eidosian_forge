from __future__ import annotations
import dataclasses
import enum
import typing
@dataclasses.dataclass(frozen=True)
class _BoxSymbolsWithDashes(_BoxSymbols):
    """Box symbols for drawing.

    Extra dashes symbols.
    """
    HORIZONTAL_4_DASHES: str
    HORIZONTAL_3_DASHES: str
    HORIZONTAL_2_DASHES: str
    VERTICAL_2_DASH: str
    VERTICAL_3_DASH: str
    VERTICAL_4_DASH: str