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
class HTTPConnectionPoolRetryTests(TestCase, FakeReactorAndConnectMixin):
    """
    L{client.HTTPConnectionPool}, by using
    L{client._RetryingHTTP11ClientProtocol}, supports retrying requests done
    against previously cached connections.
    """

    def test_onlyRetryIdempotentMethods(self):
        """
        Only GET, HEAD, OPTIONS, TRACE, DELETE methods cause a retry.
        """
        pool = client.HTTPConnectionPool(None)
        connection = client._RetryingHTTP11ClientProtocol(None, pool)
        self.assertTrue(connection._shouldRetry(b'GET', RequestNotSent(), None))
        self.assertTrue(connection._shouldRetry(b'HEAD', RequestNotSent(), None))
        self.assertTrue(connection._shouldRetry(b'OPTIONS', RequestNotSent(), None))
        self.assertTrue(connection._shouldRetry(b'TRACE', RequestNotSent(), None))
        self.assertTrue(connection._shouldRetry(b'DELETE', RequestNotSent(), None))
        self.assertFalse(connection._shouldRetry(b'POST', RequestNotSent(), None))
        self.assertFalse(connection._shouldRetry(b'MYMETHOD', RequestNotSent(), None))

    def test_onlyRetryIfNoResponseReceived(self):
        """
        Only L{RequestNotSent}, L{RequestTransmissionFailed} and
        L{ResponseNeverReceived} exceptions cause a retry.
        """
        pool = client.HTTPConnectionPool(None)
        connection = client._RetryingHTTP11ClientProtocol(None, pool)
        self.assertTrue(connection._shouldRetry(b'GET', RequestNotSent(), None))
        self.assertTrue(connection._shouldRetry(b'GET', RequestTransmissionFailed([]), None))
        self.assertTrue(connection._shouldRetry(b'GET', ResponseNeverReceived([]), None))
        self.assertFalse(connection._shouldRetry(b'GET', ResponseFailed([]), None))
        self.assertFalse(connection._shouldRetry(b'GET', ConnectionRefusedError(), None))

    def test_dontRetryIfFailedDueToCancel(self):
        """
        If a request failed due to the operation being cancelled,
        C{_shouldRetry} returns C{False} to indicate the request should not be
        retried.
        """
        pool = client.HTTPConnectionPool(None)
        connection = client._RetryingHTTP11ClientProtocol(None, pool)
        exception = ResponseNeverReceived([Failure(defer.CancelledError())])
        self.assertFalse(connection._shouldRetry(b'GET', exception, None))

    def test_retryIfFailedDueToNonCancelException(self):
        """
        If a request failed with L{ResponseNeverReceived} due to some
        arbitrary exception, C{_shouldRetry} returns C{True} to indicate the
        request should be retried.
        """
        pool = client.HTTPConnectionPool(None)
        connection = client._RetryingHTTP11ClientProtocol(None, pool)
        self.assertTrue(connection._shouldRetry(b'GET', ResponseNeverReceived([Failure(Exception())]), None))

    def test_wrappedOnPersistentReturned(self):
        """
        If L{client.HTTPConnectionPool.getConnection} returns a previously
        cached connection, it will get wrapped in a
        L{client._RetryingHTTP11ClientProtocol}.
        """
        pool = client.HTTPConnectionPool(Clock())
        protocol = StubHTTPProtocol()
        protocol.makeConnection(StringTransport())
        pool._putConnection(123, protocol)
        d = pool.getConnection(123, DummyEndpoint())

        def gotConnection(connection):
            self.assertIsInstance(connection, client._RetryingHTTP11ClientProtocol)
            self.assertIdentical(connection._clientProtocol, protocol)
        return d.addCallback(gotConnection)

    def test_notWrappedOnNewReturned(self):
        """
        If L{client.HTTPConnectionPool.getConnection} returns a new
        connection, it will be returned as is.
        """
        pool = client.HTTPConnectionPool(None)
        d = pool.getConnection(123, DummyEndpoint())

        def gotConnection(connection):
            self.assertIdentical(connection.__class__, HTTP11ClientProtocol)
        return d.addCallback(gotConnection)

    def retryAttempt(self, willWeRetry):
        """
        Fail a first request, possibly retrying depending on argument.
        """
        protocols = []

        def newProtocol():
            protocol = StubHTTPProtocol()
            protocols.append(protocol)
            return defer.succeed(protocol)
        bodyProducer = object()
        request = client.Request(b'FOO', b'/', Headers(), bodyProducer, persistent=True)
        newProtocol()
        protocol = protocols[0]
        retrier = client._RetryingHTTP11ClientProtocol(protocol, newProtocol)

        def _shouldRetry(m, e, bp):
            self.assertEqual(m, b'FOO')
            self.assertIdentical(bp, bodyProducer)
            self.assertIsInstance(e, (RequestNotSent, ResponseNeverReceived))
            return willWeRetry
        retrier._shouldRetry = _shouldRetry
        d = retrier.request(request)
        self.assertEqual(len(protocols), 1)
        self.assertEqual(len(protocols[0].requests), 1)
        protocol.requests[0][1].errback(RequestNotSent())
        return (d, protocols)

    def test_retryIfShouldRetryReturnsTrue(self):
        """
        L{client._RetryingHTTP11ClientProtocol} retries when
        L{client._RetryingHTTP11ClientProtocol._shouldRetry} returns C{True}.
        """
        d, protocols = self.retryAttempt(True)
        self.assertEqual(len(protocols), 2)
        response = object()
        protocols[1].requests[0][1].callback(response)
        return d.addCallback(self.assertIdentical, response)

    def test_dontRetryIfShouldRetryReturnsFalse(self):
        """
        L{client._RetryingHTTP11ClientProtocol} does not retry when
        L{client._RetryingHTTP11ClientProtocol._shouldRetry} returns C{False}.
        """
        d, protocols = self.retryAttempt(False)
        self.assertEqual(len(protocols), 1)
        return self.assertFailure(d, RequestNotSent)

    def test_onlyRetryWithoutBody(self):
        """
        L{_RetryingHTTP11ClientProtocol} only retries queries that don't have
        a body.

        This is an implementation restriction; if the restriction is fixed,
        this test should be removed and PUT added to list of methods that
        support retries.
        """
        pool = client.HTTPConnectionPool(None)
        connection = client._RetryingHTTP11ClientProtocol(None, pool)
        self.assertTrue(connection._shouldRetry(b'GET', RequestNotSent(), None))
        self.assertFalse(connection._shouldRetry(b'GET', RequestNotSent(), object()))

    def test_onlyRetryOnce(self):
        """
        If a L{client._RetryingHTTP11ClientProtocol} fails more than once on
        an idempotent query before a response is received, it will not retry.
        """
        d, protocols = self.retryAttempt(True)
        self.assertEqual(len(protocols), 2)
        protocols[1].requests[0][1].errback(ResponseNeverReceived([]))
        self.assertEqual(len(protocols), 2)
        return self.assertFailure(d, ResponseNeverReceived)

    def test_dontRetryIfRetryAutomaticallyFalse(self):
        """
        If L{HTTPConnectionPool.retryAutomatically} is set to C{False}, don't
        wrap connections with retrying logic.
        """
        pool = client.HTTPConnectionPool(Clock())
        pool.retryAutomatically = False
        protocol = StubHTTPProtocol()
        protocol.makeConnection(StringTransport())
        pool._putConnection(123, protocol)
        d = pool.getConnection(123, DummyEndpoint())

        def gotConnection(connection):
            self.assertIdentical(connection, protocol)
        return d.addCallback(gotConnection)

    def test_retryWithNewConnection(self):
        """
        L{client.HTTPConnectionPool} creates
        {client._RetryingHTTP11ClientProtocol} with a new connection factory
        method that creates a new connection using the same key and endpoint
        as the wrapped connection.
        """
        pool = client.HTTPConnectionPool(Clock())
        key = 123
        endpoint = DummyEndpoint()
        newConnections = []

        def newConnection(k, e):
            newConnections.append((k, e))
        pool._newConnection = newConnection
        protocol = StubHTTPProtocol()
        protocol.makeConnection(StringTransport())
        pool._putConnection(key, protocol)
        d = pool.getConnection(key, endpoint)

        def gotConnection(connection):
            self.assertIsInstance(connection, client._RetryingHTTP11ClientProtocol)
            self.assertIdentical(connection._clientProtocol, protocol)
            self.assertEqual(newConnections, [])
            connection._newConnection()
            self.assertEqual(len(newConnections), 1)
            self.assertEqual(newConnections[0][0], key)
            self.assertIdentical(newConnections[0][1], endpoint)
        return d.addCallback(gotConnection)