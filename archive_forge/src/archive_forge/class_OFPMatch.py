import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
class OFPMatch(StringifyMixin):
    """
    Flow Match Structure

    This class is implementation of the flow match structure having
    compose/query API.
    There are new API and old API for compatibility. the old API is
    supposed to be removed later.

    You can define the flow match by the keyword arguments.
    The following arguments are available.

    ================ =============== ==================================
    Argument         Value           Description
    ================ =============== ==================================
    in_port          Integer 32bit   Switch input port
    in_phy_port      Integer 32bit   Switch physical input port
    metadata         Integer 64bit   Metadata passed between tables
    eth_dst          MAC address     Ethernet destination address
    eth_src          MAC address     Ethernet source address
    eth_type         Integer 16bit   Ethernet frame type
    vlan_vid         Integer 16bit   VLAN id
    vlan_pcp         Integer 8bit    VLAN priority
    ip_dscp          Integer 8bit    IP DSCP (6 bits in ToS field)
    ip_ecn           Integer 8bit    IP ECN (2 bits in ToS field)
    ip_proto         Integer 8bit    IP protocol
    ipv4_src         IPv4 address    IPv4 source address
    ipv4_dst         IPv4 address    IPv4 destination address
    tcp_src          Integer 16bit   TCP source port
    tcp_dst          Integer 16bit   TCP destination port
    udp_src          Integer 16bit   UDP source port
    udp_dst          Integer 16bit   UDP destination port
    sctp_src         Integer 16bit   SCTP source port
    sctp_dst         Integer 16bit   SCTP destination port
    icmpv4_type      Integer 8bit    ICMP type
    icmpv4_code      Integer 8bit    ICMP code
    arp_op           Integer 16bit   ARP opcode
    arp_spa          IPv4 address    ARP source IPv4 address
    arp_tpa          IPv4 address    ARP target IPv4 address
    arp_sha          MAC address     ARP source hardware address
    arp_tha          MAC address     ARP target hardware address
    ipv6_src         IPv6 address    IPv6 source address
    ipv6_dst         IPv6 address    IPv6 destination address
    ipv6_flabel      Integer 32bit   IPv6 Flow Label
    icmpv6_type      Integer 8bit    ICMPv6 type
    icmpv6_code      Integer 8bit    ICMPv6 code
    ipv6_nd_target   IPv6 address    Target address for ND
    ipv6_nd_sll      MAC address     Source link-layer for ND
    ipv6_nd_tll      MAC address     Target link-layer for ND
    mpls_label       Integer 32bit   MPLS label
    mpls_tc          Integer 8bit    MPLS TC
    mpls_bos         Integer 8bit    MPLS BoS bit
    pbb_isid         Integer 24bit   PBB I-SID
    tunnel_id        Integer 64bit   Logical Port Metadata
    ipv6_exthdr      Integer 16bit   IPv6 Extension Header pseudo-field
    pbb_uca          Integer 8bit    PBB UCA header field
                                     (EXT-256 Old version of ONF Extension)
    tcp_flags        Integer 16bit   TCP flags
                                     (EXT-109 ONF Extension)
    actset_output    Integer 32bit   Output port from action set metadata
                                     (EXT-233 ONF Extension)
    ================ =============== ==================================

    Example::

        >>> # compose
        >>> match = parser.OFPMatch(
        ...     in_port=1,
        ...     eth_type=0x86dd,
        ...     ipv6_src=('2001:db8:bd05:1d2:288a:1fc0:1:10ee',
        ...               'ffff:ffff:ffff:ffff::'),
        ...     ipv6_dst='2001:db8:bd05:1d2:288a:1fc0:1:10ee')
        >>> # query
        >>> if 'ipv6_src' in match:
        ...     print match['ipv6_src']
        ...
        ('2001:db8:bd05:1d2:288a:1fc0:1:10ee', 'ffff:ffff:ffff:ffff::')

    .. Note::

        For the list of the supported Nicira experimenter matches,
        please refer to :ref:`os_ken.ofproto.nx_match <nx_match_structures>`.

    .. Note::

        For VLAN id match field, special values are defined in OpenFlow Spec.

        1) Packets with and without a VLAN tag

            - Example::

                match = parser.OFPMatch()

            - Packet Matching

                ====================== =====
                non-VLAN-tagged        MATCH
                VLAN-tagged(vlan_id=3) MATCH
                VLAN-tagged(vlan_id=5) MATCH
                ====================== =====

        2) Only packets without a VLAN tag

            - Example::

                match = parser.OFPMatch(vlan_vid=0x0000)

            - Packet Matching

                ====================== =====
                non-VLAN-tagged        MATCH
                VLAN-tagged(vlan_id=3)   x
                VLAN-tagged(vlan_id=5)   x
                ====================== =====

        3) Only packets with a VLAN tag regardless of its value

            - Example::

                match = parser.OFPMatch(vlan_vid=(0x1000, 0x1000))

            - Packet Matching

                ====================== =====
                non-VLAN-tagged          x
                VLAN-tagged(vlan_id=3) MATCH
                VLAN-tagged(vlan_id=5) MATCH
                ====================== =====

        4) Only packets with VLAN tag and VID equal

            - Example::

                match = parser.OFPMatch(vlan_vid=(0x1000 | 3))

            - Packet Matching

                ====================== =====
                non-VLAN-tagged          x
                VLAN-tagged(vlan_id=3) MATCH
                VLAN-tagged(vlan_id=5)   x
                ====================== =====
    """

    def __init__(self, type_=None, length=None, _ordered_fields=None, **kwargs):
        """
        You can define the flow match by the keyword arguments.
        Please refer to ofproto.oxm_types for the key which you can
        define.
        """
        super(OFPMatch, self).__init__()
        self._wc = FlowWildcards()
        self._flow = Flow()
        self.fields = []
        self.type = ofproto.OFPMT_OXM
        self.length = length
        if _ordered_fields is not None:
            assert not kwargs
            self._fields2 = _ordered_fields
        else:
            kwargs = dict((ofproto.oxm_normalize_user(k, v) for k, v in kwargs.items()))
            fields = [ofproto.oxm_from_user(k, v) for k, v in kwargs.items()]
            fields.sort(key=lambda x: x[0][0] if isinstance(x[0], tuple) else x[0])
            self._fields2 = [ofproto.oxm_to_user(n, v, m) for n, v, m in fields]

    def __getitem__(self, key):
        return dict(self._fields2)[key]

    def __contains__(self, key):
        return key in dict(self._fields2)

    def iteritems(self):
        return iter(dict(self._fields2).items())

    def items(self):
        return self._fields2

    def get(self, key, default=None):
        return dict(self._fields2).get(key, default)

    def stringify_attrs(self):
        yield ('oxm_fields', dict(self._fields2))

    def to_jsondict(self):
        """
        Returns a dict expressing the flow match.
        """
        if self._composed_with_old_api():
            o2 = OFPMatch()
            o2.fields = self.fields[:]
            buf = bytearray()
            o2.serialize(buf, 0)
            o = OFPMatch.parser(bytes(buf), 0)
        else:
            o = self
        body = {'oxm_fields': [ofproto.oxm_to_jsondict(k, uv) for k, uv in o._fields2], 'length': o.length, 'type': o.type}
        return {self.__class__.__name__: body}

    @classmethod
    def from_jsondict(cls, dict_):
        """
        Returns an object which is generated from a dict.

        Exception raises:
        KeyError -- Unknown match field is defined in dict
        """
        fields = [ofproto.oxm_from_jsondict(f) for f in dict_['oxm_fields']]
        o = OFPMatch(_ordered_fields=fields)
        buf = bytearray()
        o.serialize(buf, 0)
        return OFPMatch.parser(bytes(buf), 0)

    def __str__(self):
        if self._composed_with_old_api():
            o2 = OFPMatch()
            o2.fields = self.fields[:]
            buf = bytearray()
            o2.serialize(buf, 0)
            o = OFPMatch.parser(bytes(buf), 0)
        else:
            o = self
        return super(OFPMatch, o).__str__()
    __repr__ = __str__

    def append_field(self, header, value, mask=None):
        """
        Append a match field.

        ========= =======================================================
        Argument  Description
        ========= =======================================================
        header    match field header ID which is defined automatically in
                  ``ofproto``
        value     match field value
        mask      mask value to the match field
        ========= =======================================================

        The available ``header`` is as follows.

        ====================== ===================================
        Header ID              Description
        ====================== ===================================
        OXM_OF_IN_PORT         Switch input port
        OXM_OF_IN_PHY_PORT     Switch physical input port
        OXM_OF_METADATA        Metadata passed between tables
        OXM_OF_ETH_DST         Ethernet destination address
        OXM_OF_ETH_SRC         Ethernet source address
        OXM_OF_ETH_TYPE        Ethernet frame type
        OXM_OF_VLAN_VID        VLAN id
        OXM_OF_VLAN_PCP        VLAN priority
        OXM_OF_IP_DSCP         IP DSCP (6 bits in ToS field)
        OXM_OF_IP_ECN          IP ECN (2 bits in ToS field)
        OXM_OF_IP_PROTO        IP protocol
        OXM_OF_IPV4_SRC        IPv4 source address
        OXM_OF_IPV4_DST        IPv4 destination address
        OXM_OF_TCP_SRC         TCP source port
        OXM_OF_TCP_DST         TCP destination port
        OXM_OF_UDP_SRC         UDP source port
        OXM_OF_UDP_DST         UDP destination port
        OXM_OF_SCTP_SRC        SCTP source port
        OXM_OF_SCTP_DST        SCTP destination port
        OXM_OF_ICMPV4_TYPE     ICMP type
        OXM_OF_ICMPV4_CODE     ICMP code
        OXM_OF_ARP_OP          ARP opcode
        OXM_OF_ARP_SPA         ARP source IPv4 address
        OXM_OF_ARP_TPA         ARP target IPv4 address
        OXM_OF_ARP_SHA         ARP source hardware address
        OXM_OF_ARP_THA         ARP target hardware address
        OXM_OF_IPV6_SRC        IPv6 source address
        OXM_OF_IPV6_DST        IPv6 destination address
        OXM_OF_IPV6_FLABEL     IPv6 Flow Label
        OXM_OF_ICMPV6_TYPE     ICMPv6 type
        OXM_OF_ICMPV6_CODE     ICMPv6 code
        OXM_OF_IPV6_ND_TARGET  Target address for ND
        OXM_OF_IPV6_ND_SLL     Source link-layer for ND
        OXM_OF_IPV6_ND_TLL     Target link-layer for ND
        OXM_OF_MPLS_LABEL      MPLS label
        OXM_OF_MPLS_TC         MPLS TC
        OXM_OF_MPLS_BOS        MPLS BoS bit
        OXM_OF_PBB_ISID        PBB I-SID
        OXM_OF_TUNNEL_ID       Logical Port Metadata
        OXM_OF_IPV6_EXTHDR     IPv6 Extension Header pseudo-field
        ====================== ===================================
        """
        self.fields.append(OFPMatchField.make(header, value, mask))

    def _composed_with_old_api(self):
        return self.fields and (not self._fields2) or self._wc.__dict__ != FlowWildcards().__dict__

    def serialize(self, buf, offset):
        """
        Outputs the expression of the wire protocol of the flow match into
        the buf.
        Returns the output length.
        """
        if self._composed_with_old_api():
            return self.serialize_old(buf, offset)
        fields = [ofproto.oxm_from_user(k, uv) for k, uv in self._fields2]
        hdr_pack_str = '!HH'
        field_offset = offset + struct.calcsize(hdr_pack_str)
        for n, value, mask in fields:
            field_offset += ofproto.oxm_serialize(n, value, mask, buf, field_offset)
        length = field_offset - offset
        msg_pack_into(hdr_pack_str, buf, offset, ofproto.OFPMT_OXM, length)
        self.length = length
        pad_len = utils.round_up(length, 8) - length
        msg_pack_into('%dx' % pad_len, buf, field_offset)
        return length + pad_len

    def serialize_old(self, buf, offset):
        if hasattr(self, '_serialized'):
            raise Exception('serializing an OFPMatch composed with old API multiple times is not supported')
        self._serialized = True
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IN_PORT):
            self.append_field(ofproto.OXM_OF_IN_PORT, self._flow.in_port)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IN_PHY_PORT):
            self.append_field(ofproto.OXM_OF_IN_PHY_PORT, self._flow.in_phy_port)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_METADATA):
            if self._wc.metadata_mask == UINT64_MAX:
                header = ofproto.OXM_OF_METADATA
            else:
                header = ofproto.OXM_OF_METADATA_W
            self.append_field(header, self._flow.metadata, self._wc.metadata_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ETH_DST):
            if self._wc.dl_dst_mask:
                header = ofproto.OXM_OF_ETH_DST_W
            else:
                header = ofproto.OXM_OF_ETH_DST
            self.append_field(header, self._flow.dl_dst, self._wc.dl_dst_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ETH_SRC):
            if self._wc.dl_src_mask:
                header = ofproto.OXM_OF_ETH_SRC_W
            else:
                header = ofproto.OXM_OF_ETH_SRC
            self.append_field(header, self._flow.dl_src, self._wc.dl_src_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ETH_TYPE):
            self.append_field(ofproto.OXM_OF_ETH_TYPE, self._flow.dl_type)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_VLAN_VID):
            if self._wc.vlan_vid_mask == UINT16_MAX:
                header = ofproto.OXM_OF_VLAN_VID
            else:
                header = ofproto.OXM_OF_VLAN_VID_W
            self.append_field(header, self._flow.vlan_vid, self._wc.vlan_vid_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_VLAN_PCP):
            self.append_field(ofproto.OXM_OF_VLAN_PCP, self._flow.vlan_pcp)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IP_DSCP):
            self.append_field(ofproto.OXM_OF_IP_DSCP, self._flow.ip_dscp)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IP_ECN):
            self.append_field(ofproto.OXM_OF_IP_ECN, self._flow.ip_ecn)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IP_PROTO):
            self.append_field(ofproto.OXM_OF_IP_PROTO, self._flow.ip_proto)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IPV4_SRC):
            if self._wc.ipv4_src_mask == UINT32_MAX:
                header = ofproto.OXM_OF_IPV4_SRC
            else:
                header = ofproto.OXM_OF_IPV4_SRC_W
            self.append_field(header, self._flow.ipv4_src, self._wc.ipv4_src_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IPV4_DST):
            if self._wc.ipv4_dst_mask == UINT32_MAX:
                header = ofproto.OXM_OF_IPV4_DST
            else:
                header = ofproto.OXM_OF_IPV4_DST_W
            self.append_field(header, self._flow.ipv4_dst, self._wc.ipv4_dst_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_TCP_SRC):
            self.append_field(ofproto.OXM_OF_TCP_SRC, self._flow.tcp_src)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_TCP_DST):
            self.append_field(ofproto.OXM_OF_TCP_DST, self._flow.tcp_dst)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_UDP_SRC):
            self.append_field(ofproto.OXM_OF_UDP_SRC, self._flow.udp_src)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_UDP_DST):
            self.append_field(ofproto.OXM_OF_UDP_DST, self._flow.udp_dst)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_SCTP_SRC):
            self.append_field(ofproto.OXM_OF_SCTP_SRC, self._flow.sctp_src)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_SCTP_DST):
            self.append_field(ofproto.OXM_OF_SCTP_DST, self._flow.sctp_dst)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ICMPV4_TYPE):
            self.append_field(ofproto.OXM_OF_ICMPV4_TYPE, self._flow.icmpv4_type)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ICMPV4_CODE):
            self.append_field(ofproto.OXM_OF_ICMPV4_CODE, self._flow.icmpv4_code)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ARP_OP):
            self.append_field(ofproto.OXM_OF_ARP_OP, self._flow.arp_op)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ARP_SPA):
            if self._wc.arp_spa_mask == UINT32_MAX:
                header = ofproto.OXM_OF_ARP_SPA
            else:
                header = ofproto.OXM_OF_ARP_SPA_W
            self.append_field(header, self._flow.arp_spa, self._wc.arp_spa_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ARP_TPA):
            if self._wc.arp_tpa_mask == UINT32_MAX:
                header = ofproto.OXM_OF_ARP_TPA
            else:
                header = ofproto.OXM_OF_ARP_TPA_W
            self.append_field(header, self._flow.arp_tpa, self._wc.arp_tpa_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ARP_SHA):
            if self._wc.arp_sha_mask:
                header = ofproto.OXM_OF_ARP_SHA_W
            else:
                header = ofproto.OXM_OF_ARP_SHA
            self.append_field(header, self._flow.arp_sha, self._wc.arp_sha_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ARP_THA):
            if self._wc.arp_tha_mask:
                header = ofproto.OXM_OF_ARP_THA_W
            else:
                header = ofproto.OXM_OF_ARP_THA
            self.append_field(header, self._flow.arp_tha, self._wc.arp_tha_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IPV6_SRC):
            if len(self._wc.ipv6_src_mask):
                header = ofproto.OXM_OF_IPV6_SRC_W
            else:
                header = ofproto.OXM_OF_IPV6_SRC
            self.append_field(header, self._flow.ipv6_src, self._wc.ipv6_src_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IPV6_DST):
            if len(self._wc.ipv6_dst_mask):
                header = ofproto.OXM_OF_IPV6_DST_W
            else:
                header = ofproto.OXM_OF_IPV6_DST
            self.append_field(header, self._flow.ipv6_dst, self._wc.ipv6_dst_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IPV6_FLABEL):
            if self._wc.ipv6_flabel_mask == UINT32_MAX:
                header = ofproto.OXM_OF_IPV6_FLABEL
            else:
                header = ofproto.OXM_OF_IPV6_FLABEL_W
            self.append_field(header, self._flow.ipv6_flabel, self._wc.ipv6_flabel_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ICMPV6_TYPE):
            self.append_field(ofproto.OXM_OF_ICMPV6_TYPE, self._flow.icmpv6_type)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_ICMPV6_CODE):
            self.append_field(ofproto.OXM_OF_ICMPV6_CODE, self._flow.icmpv6_code)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IPV6_ND_TARGET):
            self.append_field(ofproto.OXM_OF_IPV6_ND_TARGET, self._flow.ipv6_nd_target)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IPV6_ND_SLL):
            self.append_field(ofproto.OXM_OF_IPV6_ND_SLL, self._flow.ipv6_nd_sll)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IPV6_ND_TLL):
            self.append_field(ofproto.OXM_OF_IPV6_ND_TLL, self._flow.ipv6_nd_tll)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_MPLS_LABEL):
            self.append_field(ofproto.OXM_OF_MPLS_LABEL, self._flow.mpls_label)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_MPLS_TC):
            self.append_field(ofproto.OXM_OF_MPLS_TC, self._flow.mpls_tc)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_MPLS_BOS):
            self.append_field(ofproto.OXM_OF_MPLS_BOS, self._flow.mpls_bos)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_PBB_ISID):
            if self._wc.pbb_isid_mask:
                header = ofproto.OXM_OF_PBB_ISID_W
            else:
                header = ofproto.OXM_OF_PBB_ISID
            self.append_field(header, self._flow.pbb_isid, self._wc.pbb_isid_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_TUNNEL_ID):
            if self._wc.tunnel_id_mask:
                header = ofproto.OXM_OF_TUNNEL_ID_W
            else:
                header = ofproto.OXM_OF_TUNNEL_ID
            self.append_field(header, self._flow.tunnel_id, self._wc.tunnel_id_mask)
        if self._wc.ft_test(ofproto.OFPXMT_OFB_IPV6_EXTHDR):
            if self._wc.ipv6_exthdr_mask:
                header = ofproto.OXM_OF_IPV6_EXTHDR_W
            else:
                header = ofproto.OXM_OF_IPV6_EXTHDR
            self.append_field(header, self._flow.ipv6_exthdr, self._wc.ipv6_exthdr_mask)
        field_offset = offset + 4
        for f in self.fields:
            f.serialize(buf, field_offset)
            field_offset += f.length
        length = field_offset - offset
        msg_pack_into('!HH', buf, offset, ofproto.OFPMT_OXM, length)
        pad_len = utils.round_up(length, 8) - length
        msg_pack_into('%dx' % pad_len, buf, field_offset)
        return length + pad_len

    @classmethod
    def parser(cls, buf, offset):
        """
        Returns an object which is generated from a buffer including the
        expression of the wire protocol of the flow match.
        """
        match = OFPMatch()
        type_, length = struct.unpack_from('!HH', buf, offset)
        match.type = type_
        match.length = length
        offset += 4
        length -= 4
        exc = None
        residue = None
        try:
            cls.parser_old(match, buf, offset, length)
        except struct.error as e:
            exc = e
        fields = []
        try:
            while length > 0:
                n, value, mask, field_len = ofproto.oxm_parse(buf, offset)
                k, uv = ofproto.oxm_to_user(n, value, mask)
                fields.append((k, uv))
                offset += field_len
                length -= field_len
        except struct.error as e:
            exc = e
            residue = buf[offset:]
        match._fields2 = fields
        if exc is not None:
            raise exception.OFPTruncatedMessage(match, residue, exc)
        return match

    @staticmethod
    def parser_old(match, buf, offset, length):
        while length > 0:
            field = OFPMatchField.parser(buf, offset)
            offset += field.length
            length -= field.length
            match.fields.append(field)

    def set_in_port(self, port):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IN_PORT)
        self._flow.in_port = port

    def set_in_phy_port(self, phy_port):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IN_PHY_PORT)
        self._flow.in_phy_port = phy_port

    def set_metadata(self, metadata):
        self.set_metadata_masked(metadata, UINT64_MAX)

    def set_metadata_masked(self, metadata, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_METADATA)
        self._wc.metadata_mask = mask
        self._flow.metadata = metadata & mask

    def set_dl_dst(self, dl_dst):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ETH_DST)
        self._flow.dl_dst = dl_dst

    def set_dl_dst_masked(self, dl_dst, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ETH_DST)
        self._wc.dl_dst_mask = mask
        self._flow.dl_dst = mac.haddr_bitand(dl_dst, mask)

    def set_dl_src(self, dl_src):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ETH_SRC)
        self._flow.dl_src = dl_src

    def set_dl_src_masked(self, dl_src, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ETH_SRC)
        self._wc.dl_src_mask = mask
        self._flow.dl_src = mac.haddr_bitand(dl_src, mask)

    def set_dl_type(self, dl_type):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ETH_TYPE)
        self._flow.dl_type = dl_type

    def set_vlan_vid_none(self):
        self._wc.ft_set(ofproto.OFPXMT_OFB_VLAN_VID)
        self._wc.vlan_vid_mask = UINT16_MAX
        self._flow.vlan_vid = ofproto.OFPVID_NONE

    def set_vlan_vid(self, vid):
        self.set_vlan_vid_masked(vid, UINT16_MAX)

    def set_vlan_vid_masked(self, vid, mask):
        vid |= ofproto.OFPVID_PRESENT
        self._wc.ft_set(ofproto.OFPXMT_OFB_VLAN_VID)
        self._wc.vlan_vid_mask = mask
        self._flow.vlan_vid = vid

    def set_vlan_pcp(self, pcp):
        self._wc.ft_set(ofproto.OFPXMT_OFB_VLAN_PCP)
        self._flow.vlan_pcp = pcp

    def set_ip_dscp(self, ip_dscp):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IP_DSCP)
        self._flow.ip_dscp = ip_dscp

    def set_ip_ecn(self, ip_ecn):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IP_ECN)
        self._flow.ip_ecn = ip_ecn

    def set_ip_proto(self, ip_proto):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IP_PROTO)
        self._flow.ip_proto = ip_proto

    def set_ipv4_src(self, ipv4_src):
        self.set_ipv4_src_masked(ipv4_src, UINT32_MAX)

    def set_ipv4_src_masked(self, ipv4_src, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV4_SRC)
        self._flow.ipv4_src = ipv4_src
        self._wc.ipv4_src_mask = mask

    def set_ipv4_dst(self, ipv4_dst):
        self.set_ipv4_dst_masked(ipv4_dst, UINT32_MAX)

    def set_ipv4_dst_masked(self, ipv4_dst, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV4_DST)
        self._flow.ipv4_dst = ipv4_dst
        self._wc.ipv4_dst_mask = mask

    def set_tcp_src(self, tcp_src):
        self._wc.ft_set(ofproto.OFPXMT_OFB_TCP_SRC)
        self._flow.tcp_src = tcp_src

    def set_tcp_dst(self, tcp_dst):
        self._wc.ft_set(ofproto.OFPXMT_OFB_TCP_DST)
        self._flow.tcp_dst = tcp_dst

    def set_udp_src(self, udp_src):
        self._wc.ft_set(ofproto.OFPXMT_OFB_UDP_SRC)
        self._flow.udp_src = udp_src

    def set_udp_dst(self, udp_dst):
        self._wc.ft_set(ofproto.OFPXMT_OFB_UDP_DST)
        self._flow.udp_dst = udp_dst

    def set_sctp_src(self, sctp_src):
        self._wc.ft_set(ofproto.OFPXMT_OFB_SCTP_SRC)
        self._flow.sctp_src = sctp_src

    def set_sctp_dst(self, sctp_dst):
        self._wc.ft_set(ofproto.OFPXMT_OFB_SCTP_DST)
        self._flow.sctp_dst = sctp_dst

    def set_icmpv4_type(self, icmpv4_type):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ICMPV4_TYPE)
        self._flow.icmpv4_type = icmpv4_type

    def set_icmpv4_code(self, icmpv4_code):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ICMPV4_CODE)
        self._flow.icmpv4_code = icmpv4_code

    def set_arp_opcode(self, arp_op):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ARP_OP)
        self._flow.arp_op = arp_op

    def set_arp_spa(self, arp_spa):
        self.set_arp_spa_masked(arp_spa, UINT32_MAX)

    def set_arp_spa_masked(self, arp_spa, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ARP_SPA)
        self._wc.arp_spa_mask = mask
        self._flow.arp_spa = arp_spa

    def set_arp_tpa(self, arp_tpa):
        self.set_arp_tpa_masked(arp_tpa, UINT32_MAX)

    def set_arp_tpa_masked(self, arp_tpa, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ARP_TPA)
        self._wc.arp_tpa_mask = mask
        self._flow.arp_tpa = arp_tpa

    def set_arp_sha(self, arp_sha):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ARP_SHA)
        self._flow.arp_sha = arp_sha

    def set_arp_sha_masked(self, arp_sha, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ARP_SHA)
        self._wc.arp_sha_mask = mask
        self._flow.arp_sha = mac.haddr_bitand(arp_sha, mask)

    def set_arp_tha(self, arp_tha):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ARP_THA)
        self._flow.arp_tha = arp_tha

    def set_arp_tha_masked(self, arp_tha, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ARP_THA)
        self._wc.arp_tha_mask = mask
        self._flow.arp_tha = mac.haddr_bitand(arp_tha, mask)

    def set_ipv6_src(self, src):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_SRC)
        self._flow.ipv6_src = src

    def set_ipv6_src_masked(self, src, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_SRC)
        self._wc.ipv6_src_mask = mask
        self._flow.ipv6_src = [x & y for x, y in zip(src, mask)]

    def set_ipv6_dst(self, dst):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_DST)
        self._flow.ipv6_dst = dst

    def set_ipv6_dst_masked(self, dst, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_DST)
        self._wc.ipv6_dst_mask = mask
        self._flow.ipv6_dst = [x & y for x, y in zip(dst, mask)]

    def set_ipv6_flabel(self, flabel):
        self.set_ipv6_flabel_masked(flabel, UINT32_MAX)

    def set_ipv6_flabel_masked(self, flabel, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_FLABEL)
        self._wc.ipv6_flabel_mask = mask
        self._flow.ipv6_flabel = flabel

    def set_icmpv6_type(self, icmpv6_type):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ICMPV6_TYPE)
        self._flow.icmpv6_type = icmpv6_type

    def set_icmpv6_code(self, icmpv6_code):
        self._wc.ft_set(ofproto.OFPXMT_OFB_ICMPV6_CODE)
        self._flow.icmpv6_code = icmpv6_code

    def set_ipv6_nd_target(self, target):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_ND_TARGET)
        self._flow.ipv6_nd_target = target

    def set_ipv6_nd_sll(self, ipv6_nd_sll):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_ND_SLL)
        self._flow.ipv6_nd_sll = ipv6_nd_sll

    def set_ipv6_nd_tll(self, ipv6_nd_tll):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_ND_TLL)
        self._flow.ipv6_nd_tll = ipv6_nd_tll

    def set_mpls_label(self, mpls_label):
        self._wc.ft_set(ofproto.OFPXMT_OFB_MPLS_LABEL)
        self._flow.mpls_label = mpls_label

    def set_mpls_tc(self, mpls_tc):
        self._wc.ft_set(ofproto.OFPXMT_OFB_MPLS_TC)
        self._flow.mpls_tc = mpls_tc

    def set_mpls_bos(self, bos):
        self._wc.ft_set(ofproto.OFPXMT_OFB_MPLS_BOS)
        self._flow.mpls_bos = bos

    def set_pbb_isid(self, isid):
        self._wc.ft_set(ofproto.OFPXMT_OFB_PBB_ISID)
        self._flow.pbb_isid = isid

    def set_pbb_isid_masked(self, isid, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_PBB_ISID)
        self._wc.pbb_isid_mask = mask
        self._flow.pbb_isid = isid

    def set_tunnel_id(self, tunnel_id):
        self._wc.ft_set(ofproto.OFPXMT_OFB_TUNNEL_ID)
        self._flow.tunnel_id = tunnel_id

    def set_tunnel_id_masked(self, tunnel_id, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_TUNNEL_ID)
        self._wc.tunnel_id_mask = mask
        self._flow.tunnel_id = tunnel_id

    def set_ipv6_exthdr(self, hdr):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_EXTHDR)
        self._flow.ipv6_exthdr = hdr

    def set_ipv6_exthdr_masked(self, hdr, mask):
        self._wc.ft_set(ofproto.OFPXMT_OFB_IPV6_EXTHDR)
        self._wc.ipv6_exthdr_mask = mask
        self._flow.ipv6_exthdr = hdr