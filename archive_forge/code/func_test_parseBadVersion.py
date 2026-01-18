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
def test_parseBadVersion(self):
    """
        L{HTTPClientParser.parseVersion} raises L{ValueError} when passed an
        unparsable version.
        """
    protocol = HTTPClientParser(None, None)
    e = BadResponseVersion
    f = protocol.parseVersion

    def checkParsing(s):
        exc = self.assertRaises(e, f, s)
        self.assertEqual(exc.data, s)
    checkParsing(b'foo')
    checkParsing(b'foo/bar/baz')
    checkParsing(b'foo/')
    checkParsing(b'foo/..')
    checkParsing(b'foo/a.b')
    checkParsing(b'foo/-1.-1')