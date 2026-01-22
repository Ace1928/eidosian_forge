from __future__ import annotations
import functools
import logging
import signal
import typing
from gi.repository import GLib
from .abstract_loop import EventLoop, ExitMainLoop
def _ignore_handler(_sig: int, _frame: FrameType | None=None) -> None:
    return None