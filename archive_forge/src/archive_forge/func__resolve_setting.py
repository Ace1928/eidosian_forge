from abc import abstractmethod
from abc import ABCMeta
import threading
import time
import uuid
def _resolve_setting(self, name=None, maxsize=None, timeout=None):
    if name is None:
        while True:
            name = str(uuid.uuid4())
            if name not in self._cache:
                break
    if name in self._cache:
        raise KeyError('cache %s already in use' % name)
    if maxsize is None:
        maxsize = self._maxsize
    if maxsize is None:
        raise ValueError('Cache must have a maxsize set')
    if timeout is None:
        timeout = self._timeout
    return (name, maxsize, timeout)