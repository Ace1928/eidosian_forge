from __future__ import annotations
import sys
import os
from contextlib import contextmanager
from typing import ContextManager, Iterator, TextIO, cast
from ..utils import DummyContext
from .base import PipeInput
from .vt100 import Vt100Input
class _Pipe:
    """Wrapper around os.pipe, that ensures we don't double close any end."""

    def __init__(self) -> None:
        self.read_fd, self.write_fd = os.pipe()
        self._read_closed = False
        self._write_closed = False

    def close_read(self) -> None:
        """Close read-end if not yet closed."""
        if self._read_closed:
            return
        os.close(self.read_fd)
        self._read_closed = True

    def close_write(self) -> None:
        """Close write-end if not yet closed."""
        if self._write_closed:
            return
        os.close(self.write_fd)
        self._write_closed = True

    def close(self) -> None:
        """Close both read and write ends."""
        self.close_read()
        self.close_write()