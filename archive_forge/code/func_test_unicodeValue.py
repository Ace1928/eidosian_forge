from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_unicodeValue(self):
    """
        Using a non-string value raises an error.
        """
    return self.assertFailure(self.proto.set(b'foo', 'bar'), ClientError)