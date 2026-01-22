import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.type_desc import TypeDisp
@BMPMessage.register_type(BMP_MSG_ROUTE_MONITORING)
class BMPRouteMonitoring(BMPPeerMessage):
    """BMP Route Monitoring Message

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    version                    Version. this packet lib defines BMP ver. 3
    len                        Length field.  Ignored when encoding.
    type                       Type field.  one of BMP\\_MSG\\_ constants.
    peer_type                  The type of the peer.
    peer_flags                 Provide more information about the peer.
    peer_distinguisher         Use for L3VPN router which can have multiple
                               instance.
    peer_address               The remote IP address associated with the TCP
                               session.
    peer_as                    The Autonomous System number of the peer.
    peer_bgp_id                The BGP Identifier of the peer
    timestamp                  The time when the encapsulated routes were
                               received.
    bgp_update                 BGP Update PDU
    ========================== ===============================================
    """

    def __init__(self, bgp_update, peer_type, is_post_policy, peer_distinguisher, peer_address, peer_as, peer_bgp_id, timestamp, version=VERSION, type_=BMP_MSG_ROUTE_MONITORING, len_=None, is_adj_rib_out=False):
        super(BMPRouteMonitoring, self).__init__(peer_type=peer_type, is_post_policy=is_post_policy, peer_distinguisher=peer_distinguisher, peer_address=peer_address, peer_as=peer_as, peer_bgp_id=peer_bgp_id, timestamp=timestamp, len_=len_, type_=type_, version=version, is_adj_rib_out=is_adj_rib_out)
        self.bgp_update = bgp_update

    @classmethod
    def parser(cls, buf):
        kwargs, buf = super(BMPRouteMonitoring, cls).parser(buf)
        bgp_update, _, buf = BGPMessage.parser(buf)
        kwargs['bgp_update'] = bgp_update
        return kwargs

    def serialize_tail(self):
        msg = super(BMPRouteMonitoring, self).serialize_tail()
        msg += self.bgp_update.serialize()
        return msg