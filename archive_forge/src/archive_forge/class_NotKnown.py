from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from twisted.internet.defer import Deferred
class NotKnown:

    def __init__(self):
        self.dependants = []
        self.resolved = 0

    def addDependant(self, mutableObject, key):
        assert not self.resolved
        self.dependants.append((mutableObject, key))
    resolvedObject = None

    def resolveDependants(self, newObject):
        self.resolved = 1
        self.resolvedObject = newObject
        for mut, key in self.dependants:
            mut[key] = newObject

    def __hash__(self):
        assert 0, 'I am not to be used as a dictionary key.'