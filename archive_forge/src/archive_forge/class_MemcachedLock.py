import random
import threading
import time
import typing
from typing import Any
from typing import Mapping
import warnings
from ..api import CacheBackend
from ..api import NO_VALUE
from ... import util
class MemcachedLock:
    """Simple distributed lock using memcached."""

    def __init__(self, client_fn, key, timeout=0):
        self.client_fn = client_fn
        self.key = '_lock' + key
        self.timeout = timeout
        self._mutex = threading.Lock()

    def acquire(self, wait=True):
        client = self.client_fn()
        i = 0
        while True:
            with self._mutex:
                if client.add(self.key, 1, self.timeout):
                    return True
                elif not wait:
                    return False
            sleep_time = ((i + 1) * random.random() + 2 ** i) / 2.5
            time.sleep(sleep_time)
            if i < 15:
                i += 1

    def locked(self):
        client = self.client_fn()
        return client.get(self.key) is not None

    def release(self):
        client = self.client_fn()
        client.delete(self.key)