from __future__ import annotations
import contextlib
import fcntl
import functools
import os
import selectors
import signal
import struct
import sys
import termios
import tty
import typing
from subprocess import PIPE, Popen
from urwid import signals
from . import _raw_display_base, escape
from .common import INPUT_DESCRIPTORS_CHANGED
def append_button(b: int) -> None:
    b |= mod
    result.extend([27, ord('['), ord('M'), b + 32, x + 32, y + 32])