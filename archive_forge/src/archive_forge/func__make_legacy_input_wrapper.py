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
def _make_legacy_input_wrapper(self, event_loop, callback):
    """
        Support old Screen classes that still have a get_input_nonblocking and expect it to work.
        """

    @functools.wraps(callback)
    def wrapper():
        if self._input_timeout:
            event_loop.remove_alarm(self._input_timeout)
            self._input_timeout = None
        timeout, keys, raw = self.get_input_nonblocking()
        if timeout is not None:
            self._input_timeout = event_loop.alarm(timeout, wrapper)
        callback(keys, raw)
    return wrapper