from errno import EPERM
from socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, AddressFamily, gaierror
from types import FunctionType
from unicodedata import normalize
from unittest import skipIf
from zope.interface import implementer, providedBy, provider
from zope.interface.interface import InterfaceClass
from zope.interface.verify import verifyClass, verifyObject
from twisted import plugins
from twisted.internet import (
from twisted.internet.abstract import isIPv6Address
from twisted.internet.address import (
from twisted.internet.endpoints import StandardErrorBehavior
from twisted.internet.error import ConnectingCancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol
from twisted.internet.stdio import PipeAddress
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import ILogObserver, globalLogPublisher
from twisted.plugin import getPlugins
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.systemd import ListenFDs
from twisted.test.iosim import connectableEndpoint, connectedServerAndClient
from twisted.trial import unittest
class WrappingFactoryTests(unittest.TestCase):
    """
    Test the behaviour of our ugly implementation detail C{_WrappingFactory}.
    """

    def test_doStart(self):
        """
        L{_WrappingFactory.doStart} passes through to the wrapped factory's
        C{doStart} method, allowing application-specific setup and logging.
        """
        factory = ClientFactory()
        wf = endpoints._WrappingFactory(factory)
        wf.doStart()
        self.assertEqual(1, factory.numPorts)

    def test_doStop(self):
        """
        L{_WrappingFactory.doStop} passes through to the wrapped factory's
        C{doStop} method, allowing application-specific cleanup and logging.
        """
        factory = ClientFactory()
        factory.numPorts = 3
        wf = endpoints._WrappingFactory(factory)
        wf.doStop()
        self.assertEqual(2, factory.numPorts)

    def test_failedBuildProtocol(self):
        """
        An exception raised in C{buildProtocol} of our wrappedFactory
        results in our C{onConnection} errback being fired.
        """

        class BogusFactory(ClientFactory):
            """
            A one off factory whose C{buildProtocol} raises an C{Exception}.
            """

            def buildProtocol(self, addr):
                raise ValueError('My protocol is poorly defined.')
        wf = endpoints._WrappingFactory(BogusFactory())
        wf.buildProtocol(None)
        d = self.assertFailure(wf._onConnection, ValueError)
        d.addCallback(lambda e: self.assertEqual(e.args, ('My protocol is poorly defined.',)))
        return d

    def test_buildNoneProtocol(self):
        """
        If the wrapped factory's C{buildProtocol} returns L{None} the
        C{onConnection} errback fires with L{error.NoProtocol}.
        """
        wrappingFactory = endpoints._WrappingFactory(NoneFactory())
        wrappingFactory.buildProtocol(None)
        self.failureResultOf(wrappingFactory._onConnection, error.NoProtocol)

    def test_buildProtocolReturnsNone(self):
        """
        If the wrapped factory's C{buildProtocol} returns L{None} then
        L{endpoints._WrappingFactory.buildProtocol} returns L{None}.
        """
        wrappingFactory = endpoints._WrappingFactory(NoneFactory())
        wrappingFactory._onConnection.addErrback(lambda reason: None)
        self.assertIsNone(wrappingFactory.buildProtocol(None))

    def test_logPrefixPassthrough(self):
        """
        If the wrapped protocol provides L{ILoggingContext}, whatever is
        returned from the wrapped C{logPrefix} method is returned from
        L{_WrappingProtocol.logPrefix}.
        """
        wf = endpoints._WrappingFactory(TestFactory())
        wp = wf.buildProtocol(None)
        self.assertEqual(wp.logPrefix(), 'A Test Protocol')

    def test_logPrefixDefault(self):
        """
        If the wrapped protocol does not provide L{ILoggingContext}, the
        wrapped protocol's class name is returned from
        L{_WrappingProtocol.logPrefix}.
        """

        class NoProtocol:
            pass
        factory = TestFactory()
        factory.protocol = NoProtocol
        wf = endpoints._WrappingFactory(factory)
        wp = wf.buildProtocol(None)
        self.assertEqual(wp.logPrefix(), 'NoProtocol')

    def test_wrappedProtocolDataReceived(self):
        """
        The wrapped C{Protocol}'s C{dataReceived} will get called when our
        C{_WrappingProtocol}'s C{dataReceived} gets called.
        """
        wf = endpoints._WrappingFactory(TestFactory())
        p = wf.buildProtocol(None)
        p.makeConnection(None)
        p.dataReceived(b'foo')
        self.assertEqual(p._wrappedProtocol.data, [b'foo'])
        p.dataReceived(b'bar')
        self.assertEqual(p._wrappedProtocol.data, [b'foo', b'bar'])

    def test_wrappedProtocolTransport(self):
        """
        Our transport is properly hooked up to the wrappedProtocol when a
        connection is made.
        """
        wf = endpoints._WrappingFactory(TestFactory())
        p = wf.buildProtocol(None)
        dummyTransport = object()
        p.makeConnection(dummyTransport)
        self.assertEqual(p.transport, dummyTransport)
        self.assertEqual(p._wrappedProtocol.transport, dummyTransport)

    def test_wrappedProtocolConnectionLost(self):
        """
        Our wrappedProtocol's connectionLost method is called when
        L{_WrappingProtocol.connectionLost} is called.
        """
        tf = TestFactory()
        wf = endpoints._WrappingFactory(tf)
        p = wf.buildProtocol(None)
        p.connectionLost('fail')
        self.assertEqual(p._wrappedProtocol.connectionsLost, ['fail'])

    def test_clientConnectionFailed(self):
        """
        Calls to L{_WrappingFactory.clientConnectionLost} should errback the
        L{_WrappingFactory._onConnection} L{Deferred}
        """
        wf = endpoints._WrappingFactory(TestFactory())
        expectedFailure = Failure(error.ConnectError(string='fail'))
        wf.clientConnectionFailed(None, expectedFailure)
        errors = []

        def gotError(f):
            errors.append(f)
        wf._onConnection.addErrback(gotError)
        self.assertEqual(errors, [expectedFailure])

    def test_wrappingProtocolFileDescriptorReceiver(self):
        """
        Our L{_WrappingProtocol} should be an L{IFileDescriptorReceiver} if the
        wrapped protocol is.
        """
        connectedDeferred = None
        applicationProtocol = TestFileDescriptorReceiverProtocol()
        wrapper = endpoints._WrappingProtocol(connectedDeferred, applicationProtocol)
        self.assertTrue(interfaces.IFileDescriptorReceiver.providedBy(wrapper))
        self.assertTrue(verifyObject(interfaces.IFileDescriptorReceiver, wrapper))

    def test_wrappingProtocolNotFileDescriptorReceiver(self):
        """
        Our L{_WrappingProtocol} does not provide L{IHalfCloseableProtocol} if
        the wrapped protocol doesn't.
        """
        tp = TestProtocol()
        p = endpoints._WrappingProtocol(None, tp)
        self.assertFalse(interfaces.IFileDescriptorReceiver.providedBy(p))

    def test_wrappedProtocolFileDescriptorReceived(self):
        """
        L{_WrappingProtocol.fileDescriptorReceived} calls the wrapped
        protocol's C{fileDescriptorReceived} method.
        """
        wrappedProtocol = TestFileDescriptorReceiverProtocol()
        wrapper = endpoints._WrappingProtocol(defer.Deferred(), wrappedProtocol)
        wrapper.makeConnection(StringTransport())
        wrapper.fileDescriptorReceived(42)
        self.assertEqual(wrappedProtocol.receivedDescriptors, [42])

    def test_wrappingProtocolHalfCloseable(self):
        """
        Our L{_WrappingProtocol} should be an L{IHalfCloseableProtocol} if the
        C{wrappedProtocol} is.
        """
        cd = object()
        hcp = TestHalfCloseableProtocol()
        p = endpoints._WrappingProtocol(cd, hcp)
        self.assertEqual(interfaces.IHalfCloseableProtocol.providedBy(p), True)

    def test_wrappingProtocolNotHalfCloseable(self):
        """
        Our L{_WrappingProtocol} should not provide L{IHalfCloseableProtocol}
        if the C{WrappedProtocol} doesn't.
        """
        tp = TestProtocol()
        p = endpoints._WrappingProtocol(None, tp)
        self.assertEqual(interfaces.IHalfCloseableProtocol.providedBy(p), False)

    def test_wrappingProtocolHandshakeListener(self):
        """
        Our L{_WrappingProtocol} should be an L{IHandshakeListener} if
        the C{wrappedProtocol} is.
        """
        handshakeListener = TestHandshakeListener()
        wrapped = endpoints._WrappingProtocol(None, handshakeListener)
        self.assertTrue(interfaces.IHandshakeListener.providedBy(wrapped))

    def test_wrappingProtocolNotHandshakeListener(self):
        """
        Our L{_WrappingProtocol} should not provide L{IHandshakeListener}
        if the C{wrappedProtocol} doesn't.
        """
        tp = TestProtocol()
        p = endpoints._WrappingProtocol(None, tp)
        self.assertFalse(interfaces.IHandshakeListener.providedBy(p))

    def test_wrappedProtocolReadConnectionLost(self):
        """
        L{_WrappingProtocol.readConnectionLost} should proxy to the wrapped
        protocol's C{readConnectionLost}
        """
        hcp = TestHalfCloseableProtocol()
        p = endpoints._WrappingProtocol(None, hcp)
        p.readConnectionLost()
        self.assertTrue(hcp.readLost)

    def test_wrappedProtocolWriteConnectionLost(self):
        """
        L{_WrappingProtocol.writeConnectionLost} should proxy to the wrapped
        protocol's C{writeConnectionLost}
        """
        hcp = TestHalfCloseableProtocol()
        p = endpoints._WrappingProtocol(None, hcp)
        p.writeConnectionLost()
        self.assertTrue(hcp.writeLost)

    def test_wrappedProtocolHandshakeCompleted(self):
        """
        L{_WrappingProtocol.handshakeCompleted} should proxy to the
        wrapped protocol's C{handshakeCompleted}
        """
        listener = TestHandshakeListener()
        wrapped = endpoints._WrappingProtocol(None, listener)
        wrapped.handshakeCompleted()
        self.assertEqual(listener.handshakeCompletedCalls, 1)