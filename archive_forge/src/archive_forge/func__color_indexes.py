from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def _color_indexes(self, color):
    indexes = self.best_match.get(color, None)
    if indexes is None:
        try:
            rgb = int(str(color), 16)
        except ValueError:
            rgb = 0
        r = rgb >> 16 & 255
        g = rgb >> 8 & 255
        b = rgb & 255
        indexes = self._closest_color(r, g, b)
        self.best_match[color] = indexes
    return indexes