from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class FilterTypeAbbreviations:
    """Table 4.44 of the 1.7 Manual (page 353ff)."""
    AHx = '/AHx'
    A85 = '/A85'
    LZW = '/LZW'
    FL = '/Fl'
    RL = '/RL'
    CCF = '/CCF'
    DCT = '/DCT'