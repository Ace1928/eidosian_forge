from contextlib import contextmanager
import curses
from curses import setupterm, tigetnum, tigetstr, tparm
from fcntl import ioctl
from six import text_type, string_types
from os import isatty, environ
import struct
import sys
from termios import TIOCGWINSZ
def _resolve_formatter(self, attr):
    """Resolve a sugary or plain capability name, color, or compound
        formatting function name into a callable capability.

        Return a ``ParametrizingString`` or a ``FormattingString``.

        """
    if attr in COLORS:
        return self._resolve_color(attr)
    elif attr in COMPOUNDABLES:
        return self._formatting_string(self._resolve_capability(attr))
    else:
        formatters = split_into_formatters(attr)
        if all((f in COMPOUNDABLES for f in formatters)):
            return self._formatting_string(u''.join((self._resolve_formatter(s) for s in formatters)))
        else:
            return ParametrizingString(self._resolve_capability(attr))