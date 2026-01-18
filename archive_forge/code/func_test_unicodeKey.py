from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_unicodeKey(self):
    """
        Using a non-string key as argument to commands raises an error.
        """
    d1 = self.assertFailure(self.proto.set('foo', b'bar'), ClientError)
    d2 = self.assertFailure(self.proto.increment('egg'), ClientError)
    d3 = self.assertFailure(self.proto.get(1), ClientError)
    d4 = self.assertFailure(self.proto.delete('bar'), ClientError)
    d5 = self.assertFailure(self.proto.append('foo', b'bar'), ClientError)
    d6 = self.assertFailure(self.proto.prepend('foo', b'bar'), ClientError)
    d7 = self.assertFailure(self.proto.getMultiple([b'egg', 1]), ClientError)
    return gatherResults([d1, d2, d3, d4, d5, d6, d7])