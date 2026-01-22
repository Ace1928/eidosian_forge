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
@OFPFlowUpdateHeader.register_flow_update_event(ofproto.OFPFME_PAUSED, ofproto.OFP_FLOW_UPDATE_PAUSED_SIZE)
@OFPFlowUpdateHeader.register_flow_update_event(ofproto.OFPFME_RESUMED, ofproto.OFP_FLOW_UPDATE_PAUSED_SIZE)
class OFPFlowUpdatePaused(OFPFlowUpdateHeader):

    @classmethod
    def parser(cls, buf, offset):
        length, event = struct.unpack_from(ofproto.OFP_FLOW_UPDATE_PAUSED_PACK_STR, buf, offset)
        assert cls.cls_flow_update_length == length
        assert cls.cls_flow_update_event == event
        return cls(length, event)