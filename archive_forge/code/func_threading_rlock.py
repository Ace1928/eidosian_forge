import contextlib
import functools
import sys
import threading
import types
import requests
from requests_mock import adapter
from requests_mock import exceptions
@contextlib.contextmanager
def threading_rlock(timeout):
    kwargs = {}
    if sys.version_info.major >= 3:
        kwargs['timeout'] = timeout
    if not _send_lock.acquire(**kwargs):
        m = 'Could not acquire threading lock - possible deadlock scenario'
        raise Exception(m)
    try:
        yield
    finally:
        _send_lock.release()