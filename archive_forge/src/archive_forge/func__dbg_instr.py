from __future__ import annotations
import curses
import sys
import typing
from contextlib import suppress
from urwid import util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, AttrSpec, BaseScreen, RealTerminal
def _dbg_instr(self):
    curses.echo()
    self.s.nodelay(0)
    curses.halfdelay(100)
    string = self.s.getstr()
    curses.noecho()
    return string