from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_timeoutCleanDeferreds(self):
    """
        C{timeoutConnection} cleans the list of commands that it fires with
        C{TimeoutError}: C{connectionLost} doesn't try to fire them again, but
        sets the disconnected state so that future commands fail with a
        C{RuntimeError}.
        """
    d1 = self.proto.get(b'foo')
    self.clock.advance(self.proto.persistentTimeOut)
    self.assertFailure(d1, TimeoutError)
    d2 = self.proto.get(b'bar')
    self.assertFailure(d2, RuntimeError)
    return gatherResults([d1, d2])