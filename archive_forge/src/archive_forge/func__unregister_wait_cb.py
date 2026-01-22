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
def _unregister_wait_cb(self, fut):
    if self._event is not None:
        _winapi.CloseHandle(self._event)
        self._event = None
        self._event_fut = None
    self._proactor._unregister(self._ov)
    self._proactor = None
    super()._unregister_wait_cb(fut)