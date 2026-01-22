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
@NXStatsReply.register_nx_stats_type()
@_set_stats_type(ofproto.NXST_AGGREGATE, NXAggregateStats)
class NXAggregateStatsReply(NXStatsReply):

    def __init__(self, datapath):
        super(NXAggregateStatsReply, self).__init__(datapath)