import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
class Fg(FgColor, Enum):
    """
    Create ANSI sequences for the 16 standard terminal foreground text colors.
    A terminal's color settings affect how these colors appear.
    To reset any foreground color, use Fg.RESET.
    """
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    LIGHT_GRAY = 37
    DARK_GRAY = 90
    LIGHT_RED = 91
    LIGHT_GREEN = 92
    LIGHT_YELLOW = 93
    LIGHT_BLUE = 94
    LIGHT_MAGENTA = 95
    LIGHT_CYAN = 96
    WHITE = 97
    RESET = 39

    def __str__(self) -> str:
        """
        Return ANSI color sequence instead of enum name
        This is helpful when using an Fg in an f-string or format() call
        e.g. my_str = f"{Fg.BLUE}hello{Fg.RESET}"
        """
        return f'{CSI}{self.value}m'