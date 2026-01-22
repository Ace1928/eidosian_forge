from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
class NoConsoleScreenBufferError(Exception):
    """
    Raised when the application is not running inside a Windows Console, but
    the user tries to instantiate Win32Output.
    """

    def __init__(self):
        xterm = 'xterm' in os.environ.get('TERM', '')
        if xterm:
            message = 'Found %s, while expecting a Windows console. Maybe try to run this program using "winpty" or run it in cmd.exe instead. Or otherwise, in case of Cygwin, use the Python executable that is compiled for Cygwin.' % os.environ['TERM']
        else:
            message = 'No Windows console found. Are you running cmd.exe?'
        super(NoConsoleScreenBufferError, self).__init__(message)