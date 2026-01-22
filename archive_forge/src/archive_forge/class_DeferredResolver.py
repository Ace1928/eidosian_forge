from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
class DeferredResolver:

    def __init__(self, resolverDeferred):
        self.waiting = []
        resolverDeferred.addCallback(self.gotRealResolver)

    def gotRealResolver(self, resolver):
        w = self.waiting
        self.__dict__ = resolver.__dict__
        self.__class__ = resolver.__class__
        for d in w:
            d.callback(resolver)

    def __getattr__(self, name):
        if name.startswith('lookup') or name in ('getHostByName', 'query'):
            self.waiting.append(defer.Deferred())
            return makePlaceholder(self.waiting[-1], name)
        raise AttributeError(name)