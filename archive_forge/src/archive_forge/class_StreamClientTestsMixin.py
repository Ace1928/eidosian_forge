import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
class StreamClientTestsMixin:
    """
    This mixin defines tests applicable to SOCK_STREAM client implementations.

    This must be mixed in to a L{ReactorBuilder
    <twisted.internet.test.reactormixins.ReactorBuilder>} subclass, as it
    depends on several of its methods.

    Then the methods C{connect} and C{listen} must defined, defining a client
    and a server communicating with each other.
    """

    def test_interface(self):
        """
        The C{connect} method returns an object providing L{IConnector}.
        """
        reactor = self.buildReactor()
        connector = self.connect(reactor, ClientFactory())
        self.assertTrue(verifyObject(IConnector, connector))

    def test_clientConnectionFailedStopsReactor(self):
        """
        The reactor can be stopped by a client factory's
        C{clientConnectionFailed} method.
        """
        reactor = self.buildReactor()
        needsRunningReactor(reactor, lambda: self.connect(reactor, Stop(reactor)))
        self.runReactor(reactor)

    def test_connectEvent(self):
        """
        This test checks that we correctly get notifications event for a
        client.  This ought to prevent a regression under Windows using the
        GTK2 reactor.  See #3925.
        """
        reactor = self.buildReactor()
        self.listen(reactor, ServerFactory.forProtocol(Protocol))
        connected = []

        class CheckConnection(Protocol):

            def connectionMade(self):
                connected.append(self)
                reactor.stop()
        clientFactory = Stop(reactor)
        clientFactory.protocol = CheckConnection
        needsRunningReactor(reactor, lambda: self.connect(reactor, clientFactory))
        reactor.run()
        self.assertTrue(connected)

    def test_unregisterProducerAfterDisconnect(self):
        """
        If a producer is unregistered from a transport after the transport has
        been disconnected (by the peer) and after C{loseConnection} has been
        called, the transport is not re-added to the reactor as a writer as
        would be necessary if the transport were still connected.
        """
        reactor = self.buildReactor()
        self.listen(reactor, ServerFactory.forProtocol(ClosingProtocol))
        finished = Deferred()
        finished.addErrback(log.err)
        finished.addCallback(lambda ign: reactor.stop())
        writing = []

        class ClientProtocol(Protocol):
            """
            Protocol to connect, register a producer, try to lose the
            connection, wait for the server to disconnect from us, and then
            unregister the producer.
            """

            def connectionMade(self):
                log.msg('ClientProtocol.connectionMade')
                self.transport.registerProducer(_SimplePullProducer(self.transport), False)
                self.transport.loseConnection()

            def connectionLost(self, reason):
                log.msg('ClientProtocol.connectionLost')
                self.unregister()
                writing.append(self.transport in _getWriters(reactor))
                finished.callback(None)

            def unregister(self):
                log.msg('ClientProtocol unregister')
                self.transport.unregisterProducer()
        clientFactory = ClientFactory()
        clientFactory.protocol = ClientProtocol
        self.connect(reactor, clientFactory)
        self.runReactor(reactor)
        self.assertFalse(writing[0], 'Transport was writing after unregisterProducer.')

    def test_disconnectWhileProducing(self):
        """
        If C{loseConnection} is called while a producer is registered with the
        transport, the connection is closed after the producer is unregistered.
        """
        reactor = self.buildReactor()
        skippedReactors = ['Glib2Reactor', 'Gtk2Reactor']
        reactorClassName = reactor.__class__.__name__
        if reactorClassName in skippedReactors and platform.isWindows():
            raise SkipTest('A pygobject/pygtk bug disables this functionality on Windows.')

        class Producer:

            def resumeProducing(self):
                log.msg('Producer.resumeProducing')
        self.listen(reactor, ServerFactory.forProtocol(Protocol))
        finished = Deferred()
        finished.addErrback(log.err)
        finished.addCallback(lambda ign: reactor.stop())

        class ClientProtocol(Protocol):
            """
            Protocol to connect, register a producer, try to lose the
            connection, unregister the producer, and wait for the connection to
            actually be lost.
            """

            def connectionMade(self):
                log.msg('ClientProtocol.connectionMade')
                self.transport.registerProducer(Producer(), False)
                self.transport.loseConnection()
                reactor.callLater(0, reactor.callLater, 0, self.unregister)

            def unregister(self):
                log.msg('ClientProtocol unregister')
                self.transport.unregisterProducer()
                reactor.callLater(1.0, finished.errback, Failure(Exception('Connection was not lost')))

            def connectionLost(self, reason):
                log.msg('ClientProtocol.connectionLost')
                finished.callback(None)
        clientFactory = ClientFactory()
        clientFactory.protocol = ClientProtocol
        self.connect(reactor, clientFactory)
        self.runReactor(reactor)