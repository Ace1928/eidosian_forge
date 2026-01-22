from twisted.internet import defer, reactor, task
from twisted.trial import unittest
class _TPF:
    stopped = False

    def __call__(self):
        return self.stopped