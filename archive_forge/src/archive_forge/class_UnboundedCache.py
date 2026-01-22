from abc import abstractmethod
from abc import ABCMeta
import threading
import time
import uuid
class UnboundedCache(Cache):
    """
    a simple unbounded cache backed by a dictionary
    """

    def __init__(self):
        self._data = dict()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def clear(self):
        self._data.clear()

    def invalidate(self, key):
        try:
            del self._data[key]
        except KeyError:
            pass

    def put(self, key, val):
        self._data[key] = val