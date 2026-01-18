from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_prepend(self):
    """
        L{MemCacheProtocol.prepend} behaves like a L{MemCacheProtocol.set}
        method: it returns a L{Deferred} which is called back with C{True} when
        the operation succeeds.
        """
    return self._test(self.proto.prepend(b'foo', b'bar'), b'prepend foo 0 0 3\r\nbar\r\n', b'STORED\r\n', True)