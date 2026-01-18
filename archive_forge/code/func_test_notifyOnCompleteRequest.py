import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
def test_notifyOnCompleteRequest(self):
    """
        A request sent to a HTTP/2 connection fires the
        L{http.Request.notifyFinish} callback with a L{None} value.
        """
    connection = H2Connection()
    connection.requestFactory = NotifyingRequestFactory(DummyHTTPHandler)
    _, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
    deferreds = connection.requestFactory.results
    self.assertEqual(len(deferreds), 1)

    def validate(result):
        self.assertIsNone(result)
    d = deferreds[0]
    d.addCallback(validate)
    cleanupCallback = connection._streamCleanupCallbacks[1]
    return defer.gatherResults([d, cleanupCallback])