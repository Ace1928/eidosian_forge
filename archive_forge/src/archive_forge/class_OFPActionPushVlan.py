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
@OFPAction.register_action_type(ofproto.OFPAT_PUSH_VLAN, ofproto.OFP_ACTION_PUSH_SIZE)
class OFPActionPushVlan(OFPAction):
    """
    Push VLAN action

    This action pushes a new VLAN tag to the packet.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    ethertype        Ether type.  The default is 802.1Q. (0x8100)
    ================ ======================================================
    """

    def __init__(self, ethertype=ether.ETH_TYPE_8021Q, type_=None, len_=None):
        super(OFPActionPushVlan, self).__init__()
        self.ethertype = ethertype

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, ethertype = struct.unpack_from(ofproto.OFP_ACTION_PUSH_PACK_STR, buf, offset)
        return cls(ethertype)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_PUSH_PACK_STR, buf, offset, self.type, self.len, self.ethertype)