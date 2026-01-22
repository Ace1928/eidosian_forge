import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.type_desc import TypeDisp
@BMPMessage.register_type(BMP_MSG_STATISTICS_REPORT)
class BMPStatisticsReport(BMPPeerMessage):
    """BMP Statistics Report Message

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
    stats                      Statistics (one or more stats encoded as a TLV)
    ========================== ===============================================
    """
    _TLV_PACK_STR = '!HH'
    _MIN_LEN = struct.calcsize(_TLV_PACK_STR)

    def __init__(self, stats, peer_type, is_post_policy, peer_distinguisher, peer_address, peer_as, peer_bgp_id, timestamp, version=VERSION, type_=BMP_MSG_STATISTICS_REPORT, len_=None, is_adj_rib_out=False):
        super(BMPStatisticsReport, self).__init__(peer_type=peer_type, is_post_policy=is_post_policy, peer_distinguisher=peer_distinguisher, peer_address=peer_address, peer_as=peer_as, peer_bgp_id=peer_bgp_id, timestamp=timestamp, len_=len_, type_=type_, version=version, is_adj_rib_out=is_adj_rib_out)
        self.stats = stats

    @classmethod
    def parser(cls, buf):
        kwargs, rest = super(BMPStatisticsReport, cls).parser(buf)
        stats_count, = struct.unpack_from('!I', bytes(rest))
        buf = rest[struct.calcsize('!I'):]
        stats = []
        while len(buf):
            if len(buf) < cls._MIN_LEN:
                raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._MIN_LEN))
            type_, len_ = struct.unpack_from(cls._TLV_PACK_STR, bytes(buf))
            if len(buf) < cls._MIN_LEN + len_:
                raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._MIN_LEN + len_))
            value = buf[cls._MIN_LEN:cls._MIN_LEN + len_]
            if type_ == BMP_STAT_TYPE_REJECTED or type_ == BMP_STAT_TYPE_DUPLICATE_PREFIX or type_ == BMP_STAT_TYPE_DUPLICATE_WITHDRAW or (type_ == BMP_STAT_TYPE_INV_UPDATE_DUE_TO_CLUSTER_LIST_LOOP) or (type_ == BMP_STAT_TYPE_INV_UPDATE_DUE_TO_AS_PATH_LOOP) or (type_ == BMP_STAT_TYPE_INV_UPDATE_DUE_TO_ORIGINATOR_ID) or (type_ == BMP_STAT_TYPE_INV_UPDATE_DUE_TO_AS_CONFED_LOOP):
                value, = struct.unpack_from('!I', bytes(value))
            elif type_ == BMP_STAT_TYPE_ADJ_RIB_IN or type_ == BMP_STAT_TYPE_LOC_RIB or type_ == BMP_STAT_TYPE_ADJ_RIB_OUT or (type_ == BMP_STAT_TYPE_EXPORT_RIB):
                value, = struct.unpack_from('!Q', bytes(value))
            buf = buf[cls._MIN_LEN + len_:]
            stats.append({'type': type_, 'len': len_, 'value': value})
        kwargs['stats'] = stats
        return kwargs

    def serialize_tail(self):
        msg = super(BMPStatisticsReport, self).serialize_tail()
        stats_count = len(self.stats)
        msg += bytearray(struct.pack('!I', stats_count))
        for v in self.stats:
            t = v['type']
            if t == BMP_STAT_TYPE_REJECTED or t == BMP_STAT_TYPE_DUPLICATE_PREFIX or t == BMP_STAT_TYPE_DUPLICATE_WITHDRAW or (t == BMP_STAT_TYPE_INV_UPDATE_DUE_TO_CLUSTER_LIST_LOOP) or (t == BMP_STAT_TYPE_INV_UPDATE_DUE_TO_AS_PATH_LOOP) or (t == BMP_STAT_TYPE_INV_UPDATE_DUE_TO_ORIGINATOR_ID) or (t == BMP_STAT_TYPE_INV_UPDATE_DUE_TO_AS_CONFED_LOOP):
                valuepackstr = 'I'
            elif t == BMP_STAT_TYPE_ADJ_RIB_IN or t == BMP_STAT_TYPE_LOC_RIB or t == BMP_STAT_TYPE_ADJ_RIB_OUT or (t == BMP_STAT_TYPE_EXPORT_RIB):
                valuepackstr = 'Q'
            else:
                continue
            v['len'] = struct.calcsize(valuepackstr)
            msg += bytearray(struct.pack(self._TLV_PACK_STR + valuepackstr, t, v['len'], v['value']))
        return msg