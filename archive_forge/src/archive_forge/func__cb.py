from PySide6.QtCore import (QCoreApplication, QDateTime, QDeadlineTimer,
from . import futures
from . import tasks
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import enum
import os
import signal
import socket
import subprocess
import typing
import warnings
@Slot()
def _cb(self) -> None:
    if self._state == QAsyncioHandle.HandleState.PENDING:
        if self._context is not None:
            self._context.run(self._callback, *self._args)
        else:
            self._callback(*self._args)
        self._state = QAsyncioHandle.HandleState.DONE