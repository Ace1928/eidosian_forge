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
@_set_nxm_headers([ofproto_v1_0.NXM_NX_ARP_SHA])
class MFArpSha(MFField):

    @classmethod
    def make(cls, header):
        return cls(header, MF_PACK_STRING_MAC)

    def put(self, buf, offset, rule):
        return self._put(buf, offset, rule.flow.arp_sha)