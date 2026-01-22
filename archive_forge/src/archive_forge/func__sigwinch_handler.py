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
def _sigwinch_handler(self, signum: int=28, frame: FrameType | None=None) -> None:
    """
        frame -- will always be None when the GLib event loop is being used.
        """
    logger = self.logger.getChild('signal_handlers')
    logger.debug(f'SIGWINCH handler called with signum={signum!r}, frame={frame!r}')
    if IS_WINDOWS or not self._resized:
        self._resize_pipe_wr.send(b'R')
        logger.debug('Sent fake resize input to the pipe')
    self._resized = True
    self.screen_buf = None