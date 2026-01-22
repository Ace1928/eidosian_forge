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
def _stop_gpm_tracking(self) -> None:
    if not self.gpm_mev:
        return
    os.kill(self.gpm_mev.pid, signal.SIGINT)
    os.waitpid(self.gpm_mev.pid, 0)
    self.gpm_mev = None