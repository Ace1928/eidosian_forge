from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Union
from ...generic import IndirectObject
from ._font_widths import STANDARD_WIDTHS
def word_width(self, word: str) -> float:
    """Sum of character widths specified in PDF font for the supplied word"""
    return sum([self.width_map.get(char, self.space_width * 2) for char in word], 0.0)