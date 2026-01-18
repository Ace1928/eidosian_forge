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
def test_receiveResponseHeaders(self):
    """
        The headers included in a response delivered to L{HTTP11ClientProtocol}
        are included on the L{Response} instance passed to the callback
        returned by the C{request} method.
        """
    d = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))

    def cbRequest(response):
        expected = Headers({b'x-foo': [b'bar', b'baz']})
        self.assertEqual(response.headers, expected)
    d.addCallback(cbRequest)
    self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nX-Foo: bar\r\nX-Foo: baz\r\n\r\n')
    return d