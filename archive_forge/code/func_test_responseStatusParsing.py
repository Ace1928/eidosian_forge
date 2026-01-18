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
def test_responseStatusParsing(self):
    """
        L{HTTPClientParser.statusReceived} parses the version, code, and phrase
        from the status line and stores them on the response object.
        """
    request = Request(b'GET', b'/', _boringHeaders, None)
    protocol = HTTPClientParser(request, None)
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
    self.assertEqual(protocol.response.version, (b'HTTP', 1, 1))
    self.assertEqual(protocol.response.code, 200)
    self.assertEqual(protocol.response.phrase, b'OK')