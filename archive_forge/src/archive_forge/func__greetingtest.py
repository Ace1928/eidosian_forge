from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def _greetingtest(self, write, testServer):
    """
        Test one of the permutations of write/writeSequence client/server.

        @param write: The name of the method to test, C{"write"} or
            C{"writeSequence"}.
        """

    class GreeteeProtocol(Protocol):
        bytes = b''

        def dataReceived(self, bytes):
            self.bytes += bytes
            if self.bytes == b'bytes':
                self.received.callback(None)

    class GreeterProtocol(Protocol):

        def connectionMade(self):
            if write == 'write':
                self.transport.write(b'bytes')
            else:
                self.transport.writeSequence([b'byt', b'es'])
    if testServer:
        server = GreeterProtocol()
        client = GreeteeProtocol()
        d = client.received = Deferred()
    else:
        server = GreeteeProtocol()
        d = server.received = Deferred()
        client = GreeterProtocol()
    loopback.loopbackAsync(server, client)
    return d