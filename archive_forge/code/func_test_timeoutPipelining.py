from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_timeoutPipelining(self):
    """
        When two requests are sent, a timeout call remains around for the
        second request, and its timeout time is correct.
        """
    d1 = self.proto.get(b'foo')
    d2 = self.proto.get(b'bar')
    d3 = Deferred()
    self.proto.connectionLost = d3.callback
    self.clock.advance(self.proto.persistentTimeOut - 1)
    self.proto.dataReceived(b'VALUE foo 0 3\r\nbar\r\nEND\r\n')

    def check(result):
        self.assertEqual(result, (0, b'bar'))
        self.assertEqual(len(self.clock.calls), 1)
        for i in range(self.proto.persistentTimeOut):
            self.clock.advance(1)
        return self.assertFailure(d2, TimeoutError).addCallback(checkTime)

    def checkTime(ignored):
        self.assertEqual(self.clock.seconds(), 2 * self.proto.persistentTimeOut - 1)
    d1.addCallback(check)
    self.assertFailure(d3, ConnectionDone)
    return d1