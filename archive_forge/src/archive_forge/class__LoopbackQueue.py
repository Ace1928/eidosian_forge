import tempfile
from zope.interface import implementer
from twisted.internet import defer, interfaces, main, protocol
from twisted.internet.interfaces import IAddress
from twisted.internet.task import deferLater
from twisted.protocols import policies
from twisted.python import failure
class _LoopbackQueue:
    """
    Trivial wrapper around a list to give it an interface like a queue, which
    the addition of also sending notifications by way of a Deferred whenever
    the list has something added to it.
    """
    _notificationDeferred = None
    disconnect = False

    def __init__(self):
        self._queue = []

    def put(self, v):
        self._queue.append(v)
        if self._notificationDeferred is not None:
            d, self._notificationDeferred = (self._notificationDeferred, None)
            d.callback(None)

    def __nonzero__(self):
        return bool(self._queue)
    __bool__ = __nonzero__

    def get(self):
        return self._queue.pop(0)