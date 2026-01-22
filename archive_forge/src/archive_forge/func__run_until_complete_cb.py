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
def _run_until_complete_cb(self, future: futures.QAsyncioFuture) -> None:
    if not future.cancelled():
        if isinstance(future.exception(), (SystemExit, KeyboardInterrupt)):
            return
    future.get_loop().stop()