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
def test_responseStatusWithoutPhrase(self):
    """
        L{HTTPClientParser.statusReceived} can parse a status line without a
        phrase (though such lines are a violation of RFC 7230, section 3.1.2;
        nevertheless some broken servers omit the phrase).
        """
    request = Request(b'GET', b'/', _boringHeaders, None)
    protocol = HTTPClientParser(request, None)
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(b'HTTP/1.1 200\r\n')
    self.assertEqual(protocol.response.version, (b'HTTP', 1, 1))
    self.assertEqual(protocol.response.code, 200)
    self.assertEqual(protocol.response.phrase, b'')