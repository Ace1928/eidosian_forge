import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_5 as ofproto
class OFPFlowDesc(StringifyMixin):

    def __init__(self, table_id=None, priority=None, idle_timeout=None, hard_timeout=None, flags=None, importance=None, cookie=None, match=None, stats=None, instructions=None, length=None):
        super(OFPFlowDesc, self).__init__()
        self.length = length
        self.table_id = table_id
        self.priority = priority
        self.idle_timeout = idle_timeout
        self.hard_timeout = hard_timeout
        self.flags = flags
        self.importance = importance
        self.cookie = cookie
        self.match = match
        self.stats = stats
        self.instructions = instructions

    @classmethod
    def parser(cls, buf, offset):
        flow_desc = cls()
        flow_desc.length, flow_desc.table_id, flow_desc.priority, flow_desc.idle_timeout, flow_desc.hard_timeout, flow_desc.flags, flow_desc.importance, flow_desc.cookie = struct.unpack_from(ofproto.OFP_FLOW_DESC_0_PACK_STR, buf, offset)
        offset += ofproto.OFP_FLOW_DESC_0_SIZE
        flow_desc.match = OFPMatch.parser(buf, offset)
        match_length = utils.round_up(flow_desc.match.length, 8)
        offset += match_length
        flow_desc.stats = OFPStats.parser(buf, offset)
        stats_length = utils.round_up(flow_desc.stats.length, 8)
        offset += stats_length
        instructions = []
        inst_length = flow_desc.length - (ofproto.OFP_FLOW_DESC_0_SIZE + match_length + stats_length)
        while inst_length > 0:
            inst = OFPInstruction.parser(buf, offset)
            instructions.append(inst)
            offset += inst.len
            inst_length -= inst.len
        flow_desc.instructions = instructions
        return flow_desc