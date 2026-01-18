from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_tooLongKey(self):
    """
        An error is raised when trying to use a too long key: the called
        command returns a L{Deferred} which fails with a L{ClientError}.
        """
    d1 = self.assertFailure(self.proto.set(b'a' * 500, b'bar'), ClientError)
    d2 = self.assertFailure(self.proto.increment(b'a' * 500), ClientError)
    d3 = self.assertFailure(self.proto.get(b'a' * 500), ClientError)
    d4 = self.assertFailure(self.proto.append(b'a' * 500, b'bar'), ClientError)
    d5 = self.assertFailure(self.proto.prepend(b'a' * 500, b'bar'), ClientError)
    d6 = self.assertFailure(self.proto.getMultiple([b'foo', b'a' * 500]), ClientError)
    return gatherResults([d1, d2, d3, d4, d5, d6])