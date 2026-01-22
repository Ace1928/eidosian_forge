from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def _erase(self, start, length):
    chars_written = c_ulong()
    self._winapi(windll.kernel32.FillConsoleOutputCharacterA, self.hconsole, c_char(b' '), DWORD(length), _coord_byval(start), byref(chars_written))
    sbinfo = self.get_win32_screen_buffer_info()
    self._winapi(windll.kernel32.FillConsoleOutputAttribute, self.hconsole, sbinfo.wAttributes, length, _coord_byval(start), byref(chars_written))