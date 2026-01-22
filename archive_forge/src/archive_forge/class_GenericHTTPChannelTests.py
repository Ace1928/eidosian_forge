import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
class GenericHTTPChannelTests(unittest.TestCase):
    """
    Tests for L{http._genericHTTPChannelProtocol}, a L{HTTPChannel}-alike which
    can handle different HTTP protocol channels.
    """
    requests = b'GET / HTTP/1.1\r\nAccept: text/html\r\nConnection: close\r\n\r\nGET / HTTP/1.0\r\n\r\n'

    def _negotiatedProtocolForTransportInstance(self, t):
        """
        Run a request using the specific instance of a transport. Returns the
        negotiated protocol string.
        """
        a = http._genericHTTPChannelProtocolFactory(b'')
        a.requestFactory = DummyHTTPHandlerProxy
        a.makeConnection(t)
        for byte in iterbytes(self.requests):
            a.dataReceived(byte)
        a.connectionLost(IOError('all done'))
        return a._negotiatedProtocol

    @skipIf(not http.H2_ENABLED, 'HTTP/2 support not present')
    def test_h2CancelsH11Timeout(self):
        """
        When the transport is switched to H2, the HTTPChannel timeouts are
        cancelled.
        """
        clock = Clock()
        a = http._genericHTTPChannelProtocolFactory(b'')
        a.requestFactory = DummyHTTPHandlerProxy
        a.timeOut = 100
        a.callLater = clock.callLater
        b = StringTransport()
        b.negotiatedProtocol = b'h2'
        a.makeConnection(b)
        hamcrest.assert_that(clock.getDelayedCalls(), hamcrest.contains(hamcrest.has_property('cancelled', hamcrest.equal_to(False))))
        h11Timeout = clock.getDelayedCalls()[0]
        a.dataReceived(b'')
        self.assertEqual(a._negotiatedProtocol, b'h2')
        self.assertTrue(h11Timeout.cancelled)
        hamcrest.assert_that(clock.getDelayedCalls(), hamcrest.contains(hamcrest.has_property('cancelled', hamcrest.equal_to(False))))

    def test_protocolUnspecified(self):
        """
        If the transport has no support for protocol negotiation (no
        negotiatedProtocol attribute), HTTP/1.1 is assumed.
        """
        b = StringTransport()
        negotiatedProtocol = self._negotiatedProtocolForTransportInstance(b)
        self.assertEqual(negotiatedProtocol, b'http/1.1')

    def test_protocolNone(self):
        """
        If the transport has no support for protocol negotiation (returns None
        for negotiatedProtocol), HTTP/1.1 is assumed.
        """
        b = StringTransport()
        b.negotiatedProtocol = None
        negotiatedProtocol = self._negotiatedProtocolForTransportInstance(b)
        self.assertEqual(negotiatedProtocol, b'http/1.1')

    def test_http11(self):
        """
        If the transport reports that HTTP/1.1 is negotiated, that's what's
        negotiated.
        """
        b = StringTransport()
        b.negotiatedProtocol = b'http/1.1'
        negotiatedProtocol = self._negotiatedProtocolForTransportInstance(b)
        self.assertEqual(negotiatedProtocol, b'http/1.1')

    @skipIf(not http.H2_ENABLED, 'HTTP/2 support not present')
    def test_http2_present(self):
        """
        If the transport reports that HTTP/2 is negotiated and HTTP/2 is
        present, that's what's negotiated.
        """
        b = StringTransport()
        b.negotiatedProtocol = b'h2'
        negotiatedProtocol = self._negotiatedProtocolForTransportInstance(b)
        self.assertEqual(negotiatedProtocol, b'h2')

    @skipIf(http.H2_ENABLED, 'HTTP/2 support present')
    def test_http2_absent(self):
        """
        If the transport reports that HTTP/2 is negotiated and HTTP/2 is not
        present, an error is encountered.
        """
        b = StringTransport()
        b.negotiatedProtocol = b'h2'
        self.assertRaises(ValueError, self._negotiatedProtocolForTransportInstance, b)

    def test_unknownProtocol(self):
        """
        If the transport reports that a protocol other than HTTP/1.1 or HTTP/2
        is negotiated, an error occurs.
        """
        b = StringTransport()
        b.negotiatedProtocol = b'smtp'
        self.assertRaises(AssertionError, self._negotiatedProtocolForTransportInstance, b)

    def test_factory(self):
        """
        The C{factory} attribute is taken from the inner channel.
        """
        a = http._genericHTTPChannelProtocolFactory(b'')
        a._channel.factory = b'Foo'
        self.assertEqual(a.factory, b'Foo')

    def test_GenericHTTPChannelPropagatesCallLater(self):
        """
        If C{callLater} is patched onto the L{http._GenericHTTPChannelProtocol}
        then we need to propagate it through to the backing channel.
        """
        clock = Clock()
        factory = http.HTTPFactory(reactor=clock)
        protocol = factory.buildProtocol(None)
        self.assertEqual(protocol.callLater, clock.callLater)
        self.assertEqual(protocol._channel.callLater, clock.callLater)

    @skipIf(not http.H2_ENABLED, 'HTTP/2 support not present')
    def test_genericHTTPChannelCallLaterUpgrade(self):
        """
        If C{callLater} is patched onto the L{http._GenericHTTPChannelProtocol}
        then we need to propagate it across onto a new backing channel after
        upgrade.
        """
        clock = Clock()
        factory = http.HTTPFactory(reactor=clock)
        protocol = factory.buildProtocol(None)
        self.assertEqual(protocol.callLater, clock.callLater)
        self.assertEqual(protocol._channel.callLater, clock.callLater)
        transport = StringTransport()
        transport.negotiatedProtocol = b'h2'
        protocol.requestFactory = DummyHTTPHandler
        protocol.makeConnection(transport)
        protocol.dataReceived(b'P')
        self.assertEqual(protocol.callLater, clock.callLater)
        self.assertEqual(protocol._channel.callLater, clock.callLater)

    @skipIf(not http.H2_ENABLED, 'HTTP/2 support not present')
    def test_unregistersProducer(self):
        """
        The L{_GenericHTTPChannelProtocol} will unregister its proxy channel
        from the transport if upgrade is negotiated.
        """
        transport = StringTransport()
        transport.negotiatedProtocol = b'h2'
        genericProtocol = http._genericHTTPChannelProtocolFactory(b'')
        genericProtocol.requestFactory = DummyHTTPHandlerProxy
        genericProtocol.makeConnection(transport)
        originalChannel = genericProtocol._channel
        self.assertIs(transport.producer, originalChannel)
        genericProtocol.dataReceived(b'P')
        self.assertIsNot(transport.producer, originalChannel)
        self.assertIs(transport.producer, genericProtocol._channel)