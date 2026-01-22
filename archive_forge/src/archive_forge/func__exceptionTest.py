from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def _exceptionTest(self, method, exceptionType, flush):

    def eb(err):
        err.trap(exceptionType)
        self.compare(err.traceback, 'Traceback unavailable\n')
        if flush:
            errs = self.flushLoggedErrors(exceptionType)
            self.assertEqual(len(errs), 1)
        return (err.type, err.value, err.traceback)
    d = self.clientFactory.getRootObject()

    def gotRootObject(root):
        d = root.callRemote(method)
        d.addErrback(eb)
        return d
    d.addCallback(gotRootObject)
    self.pump.flush()