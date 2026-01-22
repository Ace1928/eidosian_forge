from __future__ import annotations
from collections import deque
from functools import partial
from io import BytesIO
from time import time
from kombu.asynchronous.hub import READ, WRITE, Hub, get_event_loop
from kombu.exceptions import HttpError
from kombu.utils.encoding import bytes_to_str
from .base import BaseClient
def _timeout_check(self, _pycurl=pycurl):
    self._pop_from_hub()
    try:
        while 1:
            try:
                ret, _ = self._multi.socket_all()
            except pycurl.error as exc:
                ret = exc.args[0]
            if ret != _pycurl.E_CALL_MULTI_PERFORM:
                break
    finally:
        self._push_to_hub()
    self._process_pending_requests()