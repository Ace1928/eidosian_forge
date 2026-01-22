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
class DummyResponse:
    """
    Fake L{IResponse} for testing readBody that captures the protocol passed to
    deliverBody and uses it to make a connection with a transport.

    @ivar protocol: After C{deliverBody} is called, the protocol it was called
        with.

    @ivar transport: An instance created by calling C{transportFactory} which
        is used by L{DummyResponse.protocol} to make a connection.
    """
    code = 200
    phrase = b'OK'

    def __init__(self, headers=None, transportFactory=AbortableStringTransport):
        """
        @param headers: The headers for this response.  If L{None}, an empty
            L{Headers} instance will be used.
        @type headers: L{Headers}

        @param transportFactory: A callable used to construct the transport.
        """
        if headers is None:
            headers = Headers()
        self.headers = headers
        self.transport = transportFactory()

    def deliverBody(self, protocol):
        """
        Record the given protocol and use it to make a connection with
        L{DummyResponse.transport}.
        """
        self.protocol = protocol
        self.protocol.makeConnection(self.transport)