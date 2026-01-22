from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
def _resolve_color(self, color):
    """Resolve a color like red or on_bright_green into a callable
        capability."""
    color_cap = self._background_color if 'on_' in color else self._foreground_color
    offset = 8 if 'bright_' in color else 0
    base_color = color.rsplit('_', 1)[-1]
    return self._formatting_string(color_cap(getattr(curses, 'COLOR_' + base_color.upper()) + offset))