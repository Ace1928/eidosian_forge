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
class OFPFlowStatsRequestBase(OFPMultipartRequest):

    def __init__(self, datapath, flags, table_id, out_port, out_group, cookie, cookie_mask, match):
        super(OFPFlowStatsRequestBase, self).__init__(datapath, flags)
        self.table_id = table_id
        self.out_port = out_port
        self.out_group = out_group
        self.cookie = cookie
        self.cookie_mask = cookie_mask
        self.match = match

    def _serialize_stats_body(self):
        offset = ofproto.OFP_MULTIPART_REQUEST_SIZE
        msg_pack_into(ofproto.OFP_FLOW_STATS_REQUEST_0_PACK_STR, self.buf, offset, self.table_id, self.out_port, self.out_group, self.cookie, self.cookie_mask)
        offset += ofproto.OFP_FLOW_STATS_REQUEST_0_SIZE
        self.match.serialize(self.buf, offset)