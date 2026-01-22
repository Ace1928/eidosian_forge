from __future__ import annotations
import abc
import contextlib
import functools
import os
import platform
import selectors
import signal
import socket
import sys
import typing
from urwid import signals, str_util, util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, UPDATE_PALETTE_ENTRY, AttrSpec, BaseScreen, RealTerminal
def _setup_G1(self) -> None:
    """
        Initialize the G1 character set to graphics mode if required.
        """
    if self._setup_G1_done:
        return
    while True:
        with contextlib.suppress(OSError):
            self.write(escape.DESIGNATE_G1_SPECIAL)
            self.flush()
            break
    self._setup_G1_done = True