from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from twisted.internet.defer import Deferred
class _Defer(Deferred[object], NotKnown):

    def __init__(self):
        Deferred.__init__(self)
        NotKnown.__init__(self)
        self.pause()
    wasset = 0

    def __setitem__(self, n, obj):
        if self.wasset:
            raise RuntimeError('setitem should only be called once, setting {!r} to {!r}'.format(n, obj))
        else:
            self.wasset = 1
        self.callback(obj)

    def addDependant(self, dep, key):
        NotKnown.addDependant(self, dep, key)
        self.unpause()
        resovd = self.result
        self.resolveDependants(resovd)