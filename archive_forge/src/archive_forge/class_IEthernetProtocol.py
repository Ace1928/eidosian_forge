import struct
from zope.interface import Interface, implementer
from twisted.internet import protocol
from twisted.pair import raw
class IEthernetProtocol(Interface):
    """An interface for protocols that handle Ethernet frames"""

    def addProto(num, proto):
        """Add an IRawPacketProtocol protocol"""

    def datagramReceived(data, partial):
        """An Ethernet frame has been received"""