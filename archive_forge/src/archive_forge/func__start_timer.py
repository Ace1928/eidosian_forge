import asyncio
import functools
import pycares
import socket
import sys
from typing import (
from . import error
def _start_timer(self):
    timeout = self._timeout
    if timeout is None or timeout < 0 or timeout > 1:
        timeout = 1
    elif timeout == 0:
        timeout = 0.1
    self._timer = self.loop.call_later(timeout, self._timer_cb)