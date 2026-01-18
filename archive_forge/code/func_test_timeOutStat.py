from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_timeOutStat(self):
    """
        Test the timeout when stat command has started: the timeout is not
        reset until the final B{END} is received.
        """
    d1 = self.proto.stats()
    d2 = Deferred()
    self.proto.connectionLost = d2.callback
    self.proto.dataReceived(b'STAT foo bar\r\n')
    self.clock.advance(self.proto.persistentTimeOut)
    self.assertFailure(d1, TimeoutError)
    self.assertFailure(d2, ConnectionDone)
    return gatherResults([d1, d2])