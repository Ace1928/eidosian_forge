from __future__ import annotations
import curses
import sys
import typing
from contextlib import suppress
from urwid import util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, AttrSpec, BaseScreen, RealTerminal
def _getch(self, wait_tenths: int | None) -> int:
    if wait_tenths == 0:
        return self._getch_nodelay()
    if not IS_WINDOWS:
        if wait_tenths is None:
            curses.cbreak()
        else:
            curses.halfdelay(wait_tenths)
    self.s.nodelay(False)
    return self.s.getch()