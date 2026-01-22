from __future__ import annotations
import sys
import contextlib
import io
import termios
import tty
from asyncio import AbstractEventLoop, get_running_loop
from typing import Callable, ContextManager, Generator, TextIO
from ..key_binding import KeyPress
from .base import Input
from .posix_utils import PosixStdinReader
from .vt100_parser import Vt100Parser
@contextlib.contextmanager
def _attached_input(input: Vt100Input, callback: Callable[[], None]) -> Generator[None, None, None]:
    """
    Context manager that makes this input active in the current event loop.

    :param input: :class:`~prompt_toolkit.input.Input` object.
    :param callback: Called when the input is ready to read.
    """
    loop = get_running_loop()
    fd = input.fileno()
    previous = _current_callbacks.get((loop, fd))

    def callback_wrapper() -> None:
        """Wrapper around the callback that already removes the reader when
        the input is closed. Otherwise, we keep continuously calling this
        callback, until we leave the context manager (which can happen a bit
        later). This fixes issues when piping /dev/null into a prompt_toolkit
        application."""
        if input.closed:
            loop.remove_reader(fd)
        callback()
    try:
        loop.add_reader(fd, callback_wrapper)
    except PermissionError:
        raise EOFError
    _current_callbacks[loop, fd] = callback
    try:
        yield
    finally:
        loop.remove_reader(fd)
        if previous:
            loop.add_reader(fd, previous)
            _current_callbacks[loop, fd] = previous
        else:
            del _current_callbacks[loop, fd]