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
class OFPFlowMonitorRequestBase(OFPMultipartRequest):

    def __init__(self, datapath, flags, monitor_id, out_port, out_group, monitor_flags, table_id, command, match):
        super(OFPFlowMonitorRequestBase, self).__init__(datapath, flags)
        self.monitor_id = monitor_id
        self.out_port = out_port
        self.out_group = out_group
        self.monitor_flags = monitor_flags
        self.table_id = table_id
        self.command = command
        self.match = match

    def _serialize_stats_body(self):
        offset = ofproto.OFP_MULTIPART_REQUEST_SIZE
        msg_pack_into(ofproto.OFP_FLOW_MONITOR_REQUEST_0_PACK_STR, self.buf, offset, self.monitor_id, self.out_port, self.out_group, self.monitor_flags, self.table_id, self.command)
        offset += ofproto.OFP_FLOW_MONITOR_REQUEST_0_SIZE
        self.match.serialize(self.buf, offset)