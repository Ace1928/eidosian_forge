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
def _start_gpm_tracking(self) -> None:
    if not os.path.isfile('/usr/bin/mev'):
        return
    if not os.environ.get('TERM', '').lower().startswith('linux'):
        return
    m = Popen(['/usr/bin/mev', '-e', '158'], stdin=PIPE, stdout=PIPE, close_fds=True, encoding='ascii')
    fcntl.fcntl(m.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
    self.gpm_mev = m