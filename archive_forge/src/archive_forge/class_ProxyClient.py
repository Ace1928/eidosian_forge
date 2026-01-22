from twisted.internet import protocol
from twisted.python import log
class ProxyClient(Proxy):

    def connectionMade(self):
        self.peer.setPeer(self)
        self.transport.registerProducer(self.peer.transport, True)
        self.peer.transport.registerProducer(self.transport, True)
        self.peer.transport.resumeProducing()