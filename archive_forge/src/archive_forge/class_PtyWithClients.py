from __future__ import annotations
import asyncio
import codecs
import itertools
import logging
import os
import select
import signal
import warnings
from collections import deque
from concurrent import futures
from typing import TYPE_CHECKING, Any, Coroutine
from tornado.ioloop import IOLoop
class PtyWithClients:
    """A pty object with associated clients."""
    term_name: str | None

    def __init__(self, argv: Any, env: dict[str, str] | None=None, cwd: str | None=None):
        """Initialize the pty."""
        self.clients: list[Any] = []
        self.read_buffer: deque[str] = deque([], maxlen=1000)
        kwargs = {'argv': argv, 'env': env or [], 'cwd': cwd}
        if preexec_fn is not None:
            kwargs['preexec_fn'] = preexec_fn
        self.ptyproc = PtyProcessUnicode.spawn(**kwargs)
        self.ptyproc.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

    def resize_to_smallest(self) -> None:
        """Set the terminal size to that of the smallest client dimensions.

        A terminal not using the full space available is much nicer than a
        terminal trying to use more than the available space, so we keep it
        sized to the smallest client.
        """
        minrows = mincols = 10001
        for client in self.clients:
            rows, cols = client.size
            if rows is not None and rows < minrows:
                minrows = rows
            if cols is not None and cols < mincols:
                mincols = cols
        if minrows == 10001 or mincols == 10001:
            return
        rows, cols = self.ptyproc.getwinsize()
        if (rows, cols) != (minrows, mincols):
            self.ptyproc.setwinsize(minrows, mincols)

    def kill(self, sig: int=signal.SIGTERM) -> None:
        """Send a signal to the process in the pty"""
        self.ptyproc.kill(sig)

    def killpg(self, sig: int=signal.SIGTERM) -> Any:
        """Send a signal to the process group of the process in the pty"""
        if os.name == 'nt':
            return self.ptyproc.kill(sig)
        pgid = os.getpgid(self.ptyproc.pid)
        os.killpg(pgid, sig)
        return None

    async def terminate(self, force: bool=False) -> bool:
        """This forces a child process to terminate. It starts nicely with
        SIGHUP and SIGINT. If "force" is True then moves onto SIGKILL. This
        returns True if the child was terminated. This returns False if the
        child could not be terminated."""
        if os.name == 'nt':
            signals = [signal.SIGINT, signal.SIGTERM]
        else:
            signals = [signal.SIGHUP, signal.SIGCONT, signal.SIGINT, signal.SIGTERM]
        _ = IOLoop.current()

        def sleep() -> Coroutine[Any, Any, None]:
            """Sleep to allow the terminal to exit gracefully."""
            return asyncio.sleep(self.ptyproc.delayafterterminate)
        if not self.ptyproc.isalive():
            return True
        try:
            for sig in signals:
                self.kill(sig)
                await sleep()
                if not self.ptyproc.isalive():
                    return True
            if force:
                self.kill(signal.SIGKILL)
                await sleep()
                return bool(not self.ptyproc.isalive())
            return False
        except OSError:
            await sleep()
            return bool(not self.ptyproc.isalive())