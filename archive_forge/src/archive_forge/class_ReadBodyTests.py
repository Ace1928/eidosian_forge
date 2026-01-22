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
class ReadBodyTests(TestCase):
    """
    Tests for L{client.readBody}
    """

    def test_success(self):
        """
        L{client.readBody} returns a L{Deferred} which fires with the complete
        body of the L{IResponse} provider passed to it.
        """
        response = DummyResponse()
        d = client.readBody(response)
        response.protocol.dataReceived(b'first')
        response.protocol.dataReceived(b'second')
        response.protocol.connectionLost(Failure(ResponseDone()))
        self.assertEqual(self.successResultOf(d), b'firstsecond')

    def test_cancel(self):
        """
        When cancelling the L{Deferred} returned by L{client.readBody}, the
        connection to the server will be aborted.
        """
        response = DummyResponse()
        deferred = client.readBody(response)
        deferred.cancel()
        self.failureResultOf(deferred, defer.CancelledError)
        self.assertTrue(response.transport.aborting)

    def test_withPotentialDataLoss(self):
        """
        If the full body of the L{IResponse} passed to L{client.readBody} is
        not definitely received, the L{Deferred} returned by L{client.readBody}
        fires with a L{Failure} wrapping L{client.PartialDownloadError} with
        the content that was received.
        """
        response = DummyResponse()
        d = client.readBody(response)
        response.protocol.dataReceived(b'first')
        response.protocol.dataReceived(b'second')
        response.protocol.connectionLost(Failure(PotentialDataLoss()))
        failure = self.failureResultOf(d)
        failure.trap(client.PartialDownloadError)
        self.assertEqual({'status': failure.value.status, 'message': failure.value.message, 'body': failure.value.response}, {'status': b'200', 'message': b'OK', 'body': b'firstsecond'})

    def test_otherErrors(self):
        """
        If there is an exception other than L{client.PotentialDataLoss} while
        L{client.readBody} is collecting the response body, the L{Deferred}
        returned by {client.readBody} fires with that exception.
        """
        response = DummyResponse()
        d = client.readBody(response)
        response.protocol.dataReceived(b'first')
        response.protocol.connectionLost(Failure(ConnectionLost('mystery problem')))
        reason = self.failureResultOf(d)
        reason.trap(ConnectionLost)
        self.assertEqual(reason.value.args, ('mystery problem',))

    def test_deprecatedTransport(self):
        """
        Calling L{client.readBody} with a transport that does not implement
        L{twisted.internet.interfaces.ITCPTransport} produces a deprecation
        warning, but no exception when cancelling.
        """
        response = DummyResponse(transportFactory=StringTransport)
        response.transport.abortConnection = None
        d = self.assertWarns(DeprecationWarning, 'Using readBody with a transport that does not have an abortConnection method', __file__, lambda: client.readBody(response))
        d.cancel()
        self.failureResultOf(d, defer.CancelledError)

    def test_deprecatedTransportNoWarning(self):
        """
        Calling L{client.readBody} with a response that has already had its
        transport closed (eg. for a very small request) will not trigger a
        deprecation warning.
        """
        response = AlreadyCompletedDummyResponse()
        client.readBody(response)
        warnings = self.flushWarnings()
        self.assertEqual(len(warnings), 0)