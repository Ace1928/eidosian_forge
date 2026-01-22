from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
def _resolve_capability(self, atom):
    """Return a terminal code for a capname or a sugary name, or an empty
        Unicode.

        The return value is always Unicode, because otherwise it is clumsy
        (especially in Python 3) to concatenate with real (Unicode) strings.

        """
    code = tigetstr(self._sugar.get(atom, atom))
    if code:
        return code.decode('latin1')
    return u''