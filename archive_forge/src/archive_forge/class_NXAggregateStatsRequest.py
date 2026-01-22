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
class NXAggregateStatsRequest(NXStatsRequest):

    def __init__(self, datapath, flags, out_port, table_id, rule=None):
        super(NXAggregateStatsRequest, self).__init__(datapath, flags, ofproto.NXST_AGGREGATE)
        self.out_port = out_port
        self.table_id = table_id
        self.rule = rule
        self.match_len = 0

    def _serialize_vendor_stats_body(self):
        if self.rule is not None:
            offset = ofproto.NX_STATS_MSG_SIZE + ofproto.NX_AGGREGATE_STATS_REQUEST_SIZE
            self.match_len = nx_match.serialize_nxm_match(self.rule, self.buf, offset)
        msg_pack_into(ofproto.NX_AGGREGATE_STATS_REQUEST_PACK_STR, self.buf, ofproto.NX_STATS_MSG_SIZE, self.out_port, self.match_len, self.table_id)