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
def test_sendRequestHeaders(self):
    """
        L{Request.writeTo} formats header data and writes it to the given
        transport.
        """
    headers = Headers({b'x-foo': [b'bar', b'baz'], b'host': [b'example.com']})
    Request(b'GET', b'/foo', headers, None).writeTo(self.transport)
    lines = self.transport.value().split(b'\r\n')
    self.assertEqual(lines[0], b'GET /foo HTTP/1.1')
    self.assertEqual(lines[-2:], [b'', b''])
    del lines[0], lines[-2:]
    lines.sort()
    self.assertEqual(lines, [b'Connection: close', b'Host: example.com', b'X-Foo: bar', b'X-Foo: baz'])