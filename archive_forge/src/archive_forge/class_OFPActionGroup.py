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
@OFPAction.register_action_type(ofproto.OFPAT_GROUP, ofproto.OFP_ACTION_GROUP_SIZE)
class OFPActionGroup(OFPAction):
    """
    Group action

    This action indicates the group used to process the packet.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    group_id         Group identifier
    ================ ======================================================
    """

    def __init__(self, group_id=0, type_=None, len_=None):
        super(OFPActionGroup, self).__init__()
        self.group_id = group_id

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, group_id = struct.unpack_from(ofproto.OFP_ACTION_GROUP_PACK_STR, buf, offset)
        return cls(group_id)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_GROUP_PACK_STR, buf, offset, self.type, self.len, self.group_id)