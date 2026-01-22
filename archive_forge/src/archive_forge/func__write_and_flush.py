from __future__ import annotations
import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator, TextIO, cast
from .application import get_app_session, run_in_terminal
from .output import Output
def _write_and_flush(self, loop: asyncio.AbstractEventLoop | None, text: str) -> None:
    """
        Write the given text to stdout and flush.
        If an application is running, use `run_in_terminal`.
        """

    def write_and_flush() -> None:
        self._output.enable_autowrap()
        if self.raw:
            self._output.write_raw(text)
        else:
            self._output.write(text)
        self._output.flush()

    def write_and_flush_in_loop() -> None:
        run_in_terminal(write_and_flush, in_executor=False)
    if loop is None:
        write_and_flush()
    else:
        loop.call_soon_threadsafe(write_and_flush_in_loop)