from __future__ import annotations
import curses
import sys
import typing
from contextlib import suppress
from urwid import util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, AttrSpec, BaseScreen, RealTerminal
def _curs_set(self, x: int):
    if self.cursor_state in {'fixed', x}:
        return
    try:
        curses.curs_set(x)
        self.cursor_state = x
    except curses.error:
        self.cursor_state = 'fixed'