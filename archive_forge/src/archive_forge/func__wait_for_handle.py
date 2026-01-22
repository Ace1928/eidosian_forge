import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
def _wait_for_handle(self, handle, timeout, _is_cancel):
    self._check_closed()
    if timeout is None:
        ms = _winapi.INFINITE
    else:
        ms = math.ceil(timeout * 1000.0)
    ov = _overlapped.Overlapped(NULL)
    wait_handle = _overlapped.RegisterWaitWithQueue(handle, self._iocp, ov.address, ms)
    if _is_cancel:
        f = _WaitCancelFuture(ov, handle, wait_handle, loop=self._loop)
    else:
        f = _WaitHandleFuture(ov, handle, wait_handle, self, loop=self._loop)
    if f._source_traceback:
        del f._source_traceback[-1]

    def finish_wait_for_handle(trans, key, ov):
        return f._poll()
    self._cache[ov.address] = (f, ov, 0, finish_wait_for_handle)
    return f