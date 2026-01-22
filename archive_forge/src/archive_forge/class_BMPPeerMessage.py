import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.type_desc import TypeDisp
class BMPPeerMessage(BMPMessage):
    """BMP Message with Per Peer Header

    Following BMP Messages contain Per Peer Header after Common BMP Header.

    - BMP_MSG_TYPE_ROUTE_MONITRING
    - BMP_MSG_TYPE_STATISTICS_REPORT
    - BMP_MSG_PEER_UP_NOTIFICATION

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    version                    Version. this packet lib defines BMP ver. 3
    len                        Length field.  Ignored when encoding.
    type                       Type field.  one of BMP\\_MSG\\_ constants.
    peer_type                  The type of the peer.
    is_post_policy             Indicate the message reflects the post-policy
    is_adj_rib_out             Indicate the message reflects Adj-RIB-Out (defaults
                               to Adj-RIB-In)
    peer_distinguisher         Use for L3VPN router which can have multiple
                               instance.
    peer_address               The remote IP address associated with the TCP
                               session.
    peer_as                    The Autonomous System number of the peer.
    peer_bgp_id                The BGP Identifier of the peer
    timestamp                  The time when the encapsulated routes were
                               received.
    ========================== ===============================================
    """
    _PEER_HDR_PACK_STR = '!BBQ16sI4sII'
    _TYPE = {'ascii': ['peer_address', 'peer_bgp_id']}

    def __init__(self, peer_type, is_post_policy, peer_distinguisher, peer_address, peer_as, peer_bgp_id, timestamp, version=VERSION, type_=None, len_=None, is_adj_rib_out=False):
        super(BMPPeerMessage, self).__init__(version=version, len_=len_, type_=type_)
        self.peer_type = peer_type
        self.is_post_policy = is_post_policy
        self.is_adj_rib_out = is_adj_rib_out
        self.peer_distinguisher = peer_distinguisher
        self.peer_address = peer_address
        self.peer_as = peer_as
        self.peer_bgp_id = peer_bgp_id
        self.timestamp = timestamp

    @classmethod
    def parser(cls, buf):
        peer_type, peer_flags, peer_distinguisher, peer_address, peer_as, peer_bgp_id, timestamp1, timestamp2 = struct.unpack_from(cls._PEER_HDR_PACK_STR, bytes(buf))
        rest = buf[struct.calcsize(cls._PEER_HDR_PACK_STR):]
        if peer_flags & 1 << 4:
            is_adj_rib_out = True
        else:
            is_adj_rib_out = False
        if peer_flags & 1 << 6:
            is_post_policy = True
        else:
            is_post_policy = False
        if peer_flags & 1 << 7:
            peer_address = addrconv.ipv6.bin_to_text(peer_address)
        else:
            peer_address = addrconv.ipv4.bin_to_text(peer_address[-4:])
        peer_bgp_id = addrconv.ipv4.bin_to_text(peer_bgp_id)
        timestamp = float(timestamp1) + timestamp2 * 10 ** (-6)
        return ({'peer_type': peer_type, 'is_post_policy': is_post_policy, 'peer_distinguisher': peer_distinguisher, 'peer_address': peer_address, 'peer_as': peer_as, 'peer_bgp_id': peer_bgp_id, 'timestamp': timestamp, 'is_adj_rib_out': is_adj_rib_out}, rest)

    def serialize_tail(self):
        flags = 0
        if self.is_adj_rib_out:
            flags |= 1 << 4
        if self.is_post_policy:
            flags |= 1 << 6
        if ':' in self.peer_address:
            flags |= 1 << 7
            peer_address = addrconv.ipv6.text_to_bin(self.peer_address)
        else:
            peer_address = struct.pack('!12x4s', addrconv.ipv4.text_to_bin(self.peer_address))
        peer_bgp_id = addrconv.ipv4.text_to_bin(self.peer_bgp_id)
        t1, t2 = [int(t) for t in ('%.6f' % self.timestamp).split('.')]
        msg = bytearray(struct.pack(self._PEER_HDR_PACK_STR, self.peer_type, flags, self.peer_distinguisher, peer_address, self.peer_as, peer_bgp_id, t1, t2))
        return msg