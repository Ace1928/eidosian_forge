import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
class RgbFg(FgColor):
    """
    Create ANSI sequences for 24-bit (RGB) terminal foreground text colors. The terminal must support 24-bit/true-color mode.
    To reset any foreground color, including 24-bit, use Fg.RESET.
    """

    def __init__(self, r: int, g: int, b: int) -> None:
        """
        RgbFg initializer

        :param r: integer from 0-255 for the red component of the color
        :param g: integer from 0-255 for the green component of the color
        :param b: integer from 0-255 for the blue component of the color
        :raises: ValueError if r, g, or b is not in the range 0-255
        """
        if any((c < 0 or c > 255 for c in [r, g, b])):
            raise ValueError('RGB values must be integers in the range of 0 to 255')
        self._sequence = f'{CSI}38;2;{r};{g};{b}m'

    def __str__(self) -> str:
        """
        Return ANSI color sequence instead of enum name
        This is helpful when using an RgbFg in an f-string or format() call
        e.g. my_str = f"{RgbFg(0, 55, 100)}hello{Fg.RESET}"
        """
        return self._sequence