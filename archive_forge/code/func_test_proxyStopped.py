from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
def test_proxyStopped(self):
    """
        When the HTTP response parser is disconnected, the
        L{TransportProxyProducer} which was connected to it as a transport is
        stopped.
        """
    requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
    transport = self.protocol._parser.transport
    self.assertIdentical(transport._producer, self.transport)
    self.protocol._disconnectParser(Failure(ConnectionDone('connection done')))
    self.assertIdentical(transport._producer, None)
    return assertResponseFailed(self, requestDeferred, [ConnectionDone])