from zope.interface import Interface
class IRawDatagramProtocol(Interface):
    """
    An interface for protocols such as UDP, ICMP and TCP.
    """

    def addProto(num, proto):
        """
        Add a protocol on top of this one.
        """

    def datagramReceived(data, partial, source, dest, protocol, version, ihl, tos, tot_len, fragment_id, fragment_offset, dont_fragment, more_fragments, ttl):
        """
        An IP datagram has been received. Parse and process it.
        """