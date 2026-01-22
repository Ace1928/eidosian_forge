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
class TermCharset:
    __slots__ = ('_g', '_sgr_mapping', 'active', 'current')
    MAPPING: typing.ClassVar[dict[str, str | None]] = {'default': None, 'vt100': '0', 'ibmpc': 'U', 'user': None}

    def __init__(self) -> None:
        self._g = ['default', 'vt100']
        self._sgr_mapping = False
        self.active = 0
        self.current: str | None = None
        self.activate(0)

    def define(self, g: int, charset: str) -> None:
        """
        Redefine G'g' with new mapping.
        """
        self._g[g] = charset
        self.activate(g=self.active)

    def activate(self, g: int) -> None:
        """
        Activate the given charset slot.
        """
        self.active = g
        self.current = self.MAPPING.get(self._g[g], None)

    def set_sgr_ibmpc(self) -> None:
        """
        Set graphics rendition mapping to IBM PC CP437.
        """
        self._sgr_mapping = True

    def reset_sgr_ibmpc(self) -> None:
        """
        Reset graphics rendition mapping to IBM PC CP437.
        """
        self._sgr_mapping = False
        self.activate(g=self.active)

    def apply_mapping(self, char: bytes) -> bytes:
        if self._sgr_mapping or self._g[self.active] == 'ibmpc':
            dec_pos = DEC_SPECIAL_CHARS.find(char.decode('cp437'))
            if dec_pos >= 0:
                self.current = '0'
                return ALT_DEC_SPECIAL_CHARS[dec_pos].encode('cp437')
            self.current = 'U'
            return char
        return char