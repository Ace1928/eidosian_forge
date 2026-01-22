from __future__ import absolute_import
from twisted.trial.unittest import TestCase
from .._util import synchronized
class Lockable(object):

    def __init__(self):
        self._lock = FakeLock()

    @synchronized
    def check(self, x, y):
        if not self._lock.locked:
            raise RuntimeError()
        return (x, y)

    @synchronized
    def raiser(self):
        if not self._lock.locked:
            raise RuntimeError()
        raise ZeroDivisionError()