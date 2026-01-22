from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class TypArguments:
    """Table 8.2 of the PDF 1.7 reference."""
    LEFT = '/Left'
    RIGHT = '/Right'
    BOTTOM = '/Bottom'
    TOP = '/Top'