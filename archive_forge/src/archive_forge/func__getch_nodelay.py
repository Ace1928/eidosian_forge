from __future__ import annotations
import curses
import sys
import typing
from contextlib import suppress
from urwid import util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, AttrSpec, BaseScreen, RealTerminal
def _getch_nodelay(self) -> int:
    self.s.nodelay(True)
    if not IS_WINDOWS:
        while True:
            with suppress(curses.error):
                curses.cbreak()
                break
    return self.s.getch()