import struct
import base64
import netaddr
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import mac
from os_ken.lib.packet import packet
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0 as ofproto
from os_ken.ofproto import nx_actions
from os_ken import utils
import logging
class OFPActionNwAddr(OFPAction):

    def __init__(self, nw_addr):
        super(OFPActionNwAddr, self).__init__()
        if not isinstance(nw_addr, int):
            nw_addr = ip.ipv4_to_int(nw_addr)
        self.nw_addr = nw_addr

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, nw_addr = struct.unpack_from(ofproto.OFP_ACTION_NW_ADDR_PACK_STR, buf, offset)
        assert type_ in (ofproto.OFPAT_SET_NW_SRC, ofproto.OFPAT_SET_NW_DST)
        assert len_ == ofproto.OFP_ACTION_NW_ADDR_SIZE
        return cls(nw_addr)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_NW_ADDR_PACK_STR, buf, offset, self.type, self.len, self.nw_addr)

    def to_jsondict(self):
        body = {'nw_addr': ip.ipv4_to_str(self.nw_addr)}
        return {self.__class__.__name__: body}

    @classmethod
    def from_jsondict(cls, dict_):
        return cls(**dict_)