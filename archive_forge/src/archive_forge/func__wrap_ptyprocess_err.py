import os
import sys
import time
import pty
import tty
import errno
import signal
from contextlib import contextmanager
import ptyprocess
from ptyprocess.ptyprocess import use_native_pty_fork
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .spawnbase import SpawnBase
from .utils import (
@contextmanager
def _wrap_ptyprocess_err():
    """Turn ptyprocess errors into our own ExceptionPexpect errors"""
    try:
        yield
    except ptyprocess.PtyProcessError as e:
        raise ExceptionPexpect(*e.args)