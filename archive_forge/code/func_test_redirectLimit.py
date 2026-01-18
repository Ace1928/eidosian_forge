from __future__ import annotations
import zlib
from http.cookiejar import CookieJar
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
from unittest import SkipTest, skipIf
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from incremental import Version
from twisted.internet import defer, task
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import HostnameEndpoint, TCP4ClientEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import getDeprecationWarningString
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, IOPump
from twisted.test.test_sslverify import certificatesForAuthorityAndServer
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import client, error, http_headers
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web.test.injectionhelpers import (
def test_redirectLimit(self):
    """
        If the limit of redirects specified to L{client.RedirectAgent} is
        reached, the deferred fires with L{ResponseFailed} error wrapping
        a L{InfiniteRedirection} exception.
        """
    agent = self.buildAgentForWrapperTest(self.reactor)
    redirectAgent = client.RedirectAgent(agent, 1)
    deferred = redirectAgent.request(b'GET', b'http://example.com/foo')
    req, res = self.protocol.requests.pop()
    headers = http_headers.Headers({b'location': [b'http://example.com/bar']})
    response = Response((b'HTTP', 1, 1), 302, b'OK', headers, None)
    res.callback(response)
    req2, res2 = self.protocol.requests.pop()
    response2 = Response((b'HTTP', 1, 1), 302, b'OK', headers, None)
    res2.callback(response2)
    fail = self.failureResultOf(deferred, client.ResponseFailed)
    fail.value.reasons[0].trap(error.InfiniteRedirection)
    self.assertEqual(b'http://example.com/foo', fail.value.reasons[0].value.location)
    self.assertEqual(302, fail.value.response.code)