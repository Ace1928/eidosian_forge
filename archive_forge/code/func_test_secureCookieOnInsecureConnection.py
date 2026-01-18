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
def test_secureCookieOnInsecureConnection(self):
    """
        If a cookie is setup as secure, it won't be sent with the request if
        it's not over HTTPS.
        """
    uri = b'http://example.com/foo?bar'
    cookie = b'foo=1;secure'
    cookieJar = CookieJar()
    self.addCookies(cookieJar, uri, [cookie])
    self.assertEqual(len(list(cookieJar)), 1)
    agent = self.buildAgentForWrapperTest(self.reactor)
    cookieAgent = client.CookieAgent(agent, cookieJar)
    cookieAgent.request(b'GET', uri)
    req, res = self.protocol.requests.pop()
    self.assertIdentical(None, req.headers.getRawHeaders(b'cookie'))