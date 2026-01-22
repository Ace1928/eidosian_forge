import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_4 as ofproto
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_PACKET_IN_SLAVE)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_PACKET_IN_MASTER)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_PORT_STATUS_SLAVE)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_PORT_STATUS_MASTER)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_FLOW_REMOVED_SLAVE)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_FLOW_REMOVED_MASTER)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_ROLE_STATUS_SLAVE)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_ROLE_STATUS_MASTER)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_TABLE_STATUS_SLAVE)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_TABLE_STATUS_MASTER)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_REQUESTFORWARD_SLAVE)
@OFPAsyncConfigProp.register_type(ofproto.OFPACPT_REQUESTFORWARD_MASTER)
class OFPAsyncConfigPropReasons(OFPAsyncConfigProp):

    def __init__(self, type_=None, length=None, mask=None):
        self.type = type_
        self.length = length
        self.mask = mask

    @classmethod
    def parser(cls, buf):
        reasons = cls()
        reasons.type, reasons.length, reasons.mask = struct.unpack_from(ofproto.OFP_ASYNC_CONFIG_PROP_REASONS_PACK_STR, buf, 0)
        return reasons

    def serialize(self):
        self.length = ofproto.OFP_ASYNC_CONFIG_PROP_REASONS_SIZE
        buf = bytearray()
        msg_pack_into(ofproto.OFP_ASYNC_CONFIG_PROP_REASONS_PACK_STR, buf, 0, self.type, self.length, self.mask)
        return buf