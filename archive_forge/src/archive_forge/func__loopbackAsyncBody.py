import tempfile
from zope.interface import implementer
from twisted.internet import defer, interfaces, main, protocol
from twisted.internet.interfaces import IAddress
from twisted.internet.task import deferLater
from twisted.protocols import policies
from twisted.python import failure
def _loopbackAsyncBody(server, serverToClient, client, clientToServer, pumpPolicy):
    """
    Transfer bytes from the output queue of each protocol to the input of the other.

    @param server: The protocol instance representing the server-side of this
    connection.

    @param serverToClient: The L{_LoopbackQueue} holding the server's output.

    @param client: The protocol instance representing the client-side of this
    connection.

    @param clientToServer: The L{_LoopbackQueue} holding the client's output.

    @param pumpPolicy: See L{loopbackAsync}.

    @return: A L{Deferred} which fires when the connection has been closed and
        both sides have received notification of this.
    """

    def pump(source, q, target):
        sent = False
        if q:
            pumpPolicy(q, target)
            sent = True
        if sent and (not q):
            source.transport._pollProducer()
        return sent
    while 1:
        disconnect = clientSent = serverSent = False
        serverSent = pump(server, serverToClient, client)
        clientSent = pump(client, clientToServer, server)
        if not clientSent and (not serverSent):
            d = defer.Deferred()
            clientToServer._notificationDeferred = d
            serverToClient._notificationDeferred = d
            d.addCallback(_loopbackAsyncContinue, server, serverToClient, client, clientToServer, pumpPolicy)
            return d
        if serverToClient.disconnect:
            disconnect = True
            pump(server, serverToClient, client)
        elif clientToServer.disconnect:
            disconnect = True
            pump(client, clientToServer, server)
        if disconnect:
            server.connectionLost(failure.Failure(main.CONNECTION_DONE))
            client.connectionLost(failure.Failure(main.CONNECTION_DONE))
            return defer.succeed(None)