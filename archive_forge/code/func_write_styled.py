import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
def write_styled(self, text: str, style: Style) -> None:
    """Write styled text to the terminal.

        Args:
            text (str): The text to write
            style (Style): The style of the text
        """
    color = style.color
    bgcolor = style.bgcolor
    if style.reverse:
        color, bgcolor = (bgcolor, color)
    if color:
        fore = color.downgrade(ColorSystem.WINDOWS).number
        fore = fore if fore is not None else 7
        if style.bold:
            fore = fore | self.BRIGHT_BIT
        if style.dim:
            fore = fore & ~self.BRIGHT_BIT
        fore = self.ANSI_TO_WINDOWS[fore]
    else:
        fore = self._default_fore
    if bgcolor:
        back = bgcolor.downgrade(ColorSystem.WINDOWS).number
        back = back if back is not None else 0
        back = self.ANSI_TO_WINDOWS[back]
    else:
        back = self._default_back
    assert fore is not None
    assert back is not None
    SetConsoleTextAttribute(self._handle, attributes=ctypes.c_ushort(fore | back << 4))
    self.write_text(text)
    SetConsoleTextAttribute(self._handle, attributes=self._default_text)