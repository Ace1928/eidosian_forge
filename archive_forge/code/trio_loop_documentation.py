from __future__ import annotations
import logging
import typing
import exceptiongroup
import trio
from .abstract_loop import EventLoop, ExitMainLoop
Asynchronous task that watches the given file descriptor and calls
        the given callback whenever the file descriptor becomes readable.

        Parameters:
            scope: the cancellation scope that can be used to cancel the task
            fd: the file descriptor to watch
            callback: the callback to call
        