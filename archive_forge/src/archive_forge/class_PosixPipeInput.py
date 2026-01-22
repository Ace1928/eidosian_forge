from __future__ import annotations
import sys
import os
from contextlib import contextmanager
from typing import ContextManager, Iterator, TextIO, cast
from ..utils import DummyContext
from .base import PipeInput
from .vt100 import Vt100Input
class PosixPipeInput(Vt100Input, PipeInput):
    """
    Input that is send through a pipe.
    This is useful if we want to send the input programmatically into the
    application. Mostly useful for unit testing.

    Usage::

        with PosixPipeInput.create() as input:
            input.send_text('inputdata')
    """
    _id = 0

    def __init__(self, _pipe: _Pipe, _text: str='') -> None:
        self.pipe = _pipe

        class Stdin:
            encoding = 'utf-8'

            def isatty(stdin) -> bool:
                return True

            def fileno(stdin) -> int:
                return self.pipe.read_fd
        super().__init__(cast(TextIO, Stdin()))
        self.send_text(_text)
        self.__class__._id += 1
        self._id = self.__class__._id

    @classmethod
    @contextmanager
    def create(cls, text: str='') -> Iterator[PosixPipeInput]:
        pipe = _Pipe()
        try:
            yield PosixPipeInput(_pipe=pipe, _text=text)
        finally:
            pipe.close()

    def send_bytes(self, data: bytes) -> None:
        os.write(self.pipe.write_fd, data)

    def send_text(self, data: str) -> None:
        """Send text to the input."""
        os.write(self.pipe.write_fd, data.encode('utf-8'))

    def raw_mode(self) -> ContextManager[None]:
        return DummyContext()

    def cooked_mode(self) -> ContextManager[None]:
        return DummyContext()

    def close(self) -> None:
        """Close pipe fds."""
        self.pipe.close_write()

    def typeahead_hash(self) -> str:
        """
        This needs to be unique for every `PipeInput`.
        """
        return f'pipe-input-{self._id}'