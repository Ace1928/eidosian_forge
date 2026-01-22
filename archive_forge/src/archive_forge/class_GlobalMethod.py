from __future__ import annotations
import threading
from functools import partial
from dask import config
class GlobalMethod:

    def __init__(self, default, key, falsey=None):
        self._default = default
        self._key = key
        self._falsey = falsey

    def __get__(self, instance, owner=None):
        if self._key in _globals:
            if _globals[self._key]:
                return _globals[self._key]
            elif self._falsey is not None:
                return self._falsey
        return self._default