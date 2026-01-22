from random import randrange
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.address import IPv4Address
from twisted.internet.defer import succeed
from twisted.internet.interfaces import IReactorUDP, IUDPTransport
from twisted.internet.task import Clock
@implementer(IReactorUDP)
class MemoryReactor(Clock):
    """
    An L{IReactorTime} and L{IReactorUDP} provider.

    Time is controlled deterministically via the base class, L{Clock}.  UDP is
    handled in-memory by connecting protocols to instances of
    L{MemoryDatagramTransport}.

    @ivar udpPorts: A C{dict} mapping port numbers to instances of
        L{MemoryDatagramTransport}.
    """

    def __init__(self):
        Clock.__init__(self)
        self.udpPorts = {}

    def listenUDP(self, port, protocol, interface='', maxPacketSize=8192):
        """
        Pretend to bind a UDP port and connect the given protocol to it.
        """
        if port == 0:
            while True:
                port = randrange(1, 2 ** 16)
                if port not in self.udpPorts:
                    break
        if port in self.udpPorts:
            raise ValueError('Address in use')
        transport = MemoryDatagramTransport((interface, port), protocol, maxPacketSize)
        self.udpPorts[port] = transport
        protocol.makeConnection(transport)
        return transport