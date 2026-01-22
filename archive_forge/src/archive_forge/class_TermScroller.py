from __future__ import annotations
import atexit
import copy
import errno
import fcntl
import os
import pty
import selectors
import signal
import struct
import sys
import termios
import time
import traceback
import typing
import warnings
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from urwid import event_loop, util
from urwid.canvas import Canvas
from urwid.display import AttrSpec, RealTerminal
from urwid.display.escape import ALT_DEC_SPECIAL_CHARS, DEC_SPECIAL_CHARS
from urwid.widget import Sizing, Widget
from .display.common import _BASIC_COLORS, _color_desc_256, _color_desc_true
class TermScroller(list):
    """
    List subclass that handles the terminal scrollback buffer,
    truncating it as necessary.
    """
    SCROLLBACK_LINES = 10000

    def __init__(self, iterable: Iterable[typing.Any]) -> None:
        warnings.warn('`TermScroller` is deprecated. Please use `collections.deque` with non-zero `maxlen` instead.', DeprecationWarning, stacklevel=3)
        super().__init__(iterable)

    def trunc(self) -> None:
        if len(self) >= self.SCROLLBACK_LINES:
            self.pop(0)

    def append(self, obj) -> None:
        self.trunc()
        super().append(obj)

    def insert(self, idx: typing.SupportsIndex, obj) -> None:
        self.trunc()
        super().insert(idx, obj)

    def extend(self, seq) -> None:
        self.trunc()
        super().extend(seq)