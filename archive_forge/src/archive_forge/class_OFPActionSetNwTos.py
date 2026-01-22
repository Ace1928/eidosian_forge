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
@OFPAction.register_action_type(ofproto.OFPAT_SET_NW_TOS, ofproto.OFP_ACTION_NW_TOS_SIZE)
class OFPActionSetNwTos(OFPAction):
    """
    Set the IP ToS action

    This action indicates the IP ToS (DSCP field, 6 bits) to be set.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    tos              IP ToS (DSCP field, 6 bits).
    ================ ======================================================
    """

    def __init__(self, tos):
        super(OFPActionSetNwTos, self).__init__()
        self.tos = tos

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, tos = struct.unpack_from(ofproto.OFP_ACTION_NW_TOS_PACK_STR, buf, offset)
        assert type_ == ofproto.OFPAT_SET_NW_TOS
        assert len_ == ofproto.OFP_ACTION_NW_TOS_SIZE
        return cls(tos)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_NW_TOS_PACK_STR, buf, offset, self.type, self.len, self.tos)