import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
@comparable
class RemoteCacheMethod:
    """A method on a reference to a L{RemoteCache}."""

    def __init__(self, name, broker, cached, perspective):
        """(internal) initialize."""
        self.name = name
        self.broker = broker
        self.perspective = perspective
        self.cached = cached

    def __cmp__(self, other):
        return cmp((self.name, self.broker, self.perspective, self.cached), other)

    def __hash__(self):
        return hash((self.name, self.broker, self.perspective, self.cached))

    def __call__(self, *args, **kw):
        """(internal) action method."""
        cacheID = self.broker.cachedRemotelyAs(self.cached)
        if cacheID is None:
            from twisted.spread.pb import ProtocolError
            raise ProtocolError("You can't call a cached method when the object hasn't been given to the peer yet.")
        return self.broker._sendMessage(b'cache', self.perspective, cacheID, self.name, args, kw)