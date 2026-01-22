from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from twisted.internet.defer import Deferred
def addDependant(self, dep, key):
    NotKnown.addDependant(self, dep, key)
    self.unpause()
    resovd = self.result
    self.resolveDependants(resovd)