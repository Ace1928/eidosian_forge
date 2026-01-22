from __future__ import annotations
import contextlib
import functools
import logging
import selectors
import socket
import sys
import threading
import typing
from ctypes import byref
from ctypes.wintypes import DWORD
from urwid import signals
from . import _raw_display_base, _win32, escape
from .common import INPUT_DESCRIPTORS_CHANGED
def _read_raw_input(self, timeout: int) -> bytearray:
    ready = self._wait_for_input_ready(timeout)
    fd = self._input_fileno()
    chars = bytearray()
    if fd is None or fd not in ready:
        return chars
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        input_ready = selector.select(0)
        while input_ready:
            chars.extend(self._term_input_file.recv(1024))
            input_ready = selector.select(0)
        return chars