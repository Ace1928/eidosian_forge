import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
@_register_make
@_set_nxm_headers([ofproto_v1_0.NXM_NX_IPV6_DST, ofproto_v1_0.NXM_NX_IPV6_DST_W])
@MFField.register_field_header([ofproto_v1_0.NXM_NX_IPV6_DST, ofproto_v1_0.NXM_NX_IPV6_DST_W])
class MFIPV6Dst(MFIPV6, MFField):

    def __init__(self, header, value, mask=None):
        super(MFIPV6Dst, self).__init__(header, MFIPV6Dst.pack_str)
        self.value = value
        self.mask = mask

    @classmethod
    def make(cls, header):
        return cls(header, cls.pack_str)

    def put(self, buf, offset, rule):
        return self.putv6(buf, offset, rule.flow.ipv6_dst, rule.wc.ipv6_dst_mask)