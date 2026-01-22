from __future__ import annotations
import curses
import sys
import typing
from contextlib import suppress
from urwid import util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, AttrSpec, BaseScreen, RealTerminal
def _dbg_query(self, question):
    self._dbg_out(question)
    return self._dbg_instr()