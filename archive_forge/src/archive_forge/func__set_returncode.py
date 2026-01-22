import asyncio
import os
import multiprocessing
import signal
import subprocess
import sys
import time
from binascii import hexlify
from tornado.concurrent import (
from tornado import ioloop
from tornado.iostream import PipeIOStream
from tornado.log import gen_log
import typing
from typing import Optional, Any, Callable
def _set_returncode(self, status: int) -> None:
    if sys.platform == 'win32':
        self.returncode = -1
    elif os.WIFSIGNALED(status):
        self.returncode = -os.WTERMSIG(status)
    else:
        assert os.WIFEXITED(status)
        self.returncode = os.WEXITSTATUS(status)
    self.proc.returncode = self.returncode
    if self._exit_callback:
        callback = self._exit_callback
        self._exit_callback = None
        callback(self.returncode)