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