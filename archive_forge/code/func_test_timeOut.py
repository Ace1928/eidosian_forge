from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_timeOut(self):
    """
        Test the timeout on outgoing requests: when timeout is detected, all
        current commands fail with a L{TimeoutError}, and the connection is
        closed.
        """
    d1 = self.proto.get(b'foo')
    d2 = self.proto.get(b'bar')
    d3 = Deferred()
    self.proto.connectionLost = d3.callback
    self.clock.advance(self.proto.persistentTimeOut)
    self.assertFailure(d1, TimeoutError)
    self.assertFailure(d2, TimeoutError)

    def checkMessage(error):
        self.assertEqual(str(error), 'Connection timeout')
    d1.addCallback(checkMessage)
    self.assertFailure(d3, ConnectionDone)
    return gatherResults([d1, d2, d3])