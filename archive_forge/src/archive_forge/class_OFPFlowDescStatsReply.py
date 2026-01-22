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
@OFPMultipartReply.register_stats_type()
@_set_stats_type(ofproto.OFPMP_FLOW_DESC, OFPFlowDesc)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPFlowDescStatsReply(OFPMultipartReply):
    """
    Individual flow descriptions reply message

    The switch responds with this message to an individual flow descriptions
    request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPFlowDesc`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPFlowDescStatsReply, MAIN_DISPATCHER)
        def flow_desc_reply_handler(self, ev):
            flows = []
            for stat in ev.msg.body:
                flows.append('table_id=%s priority=%d '
                             'idle_timeout=%d hard_timeout=%d flags=0x%04x '
                             'importance=%d cookie=%d match=%s '
                             'stats=%s instructions=%s' %
                             (stat.table_id, stat.priority,
                              stat.idle_timeout, stat.hard_timeout,
                              stat.flags, stat.importance,
                              stat.cookie, stat.match,
                              stat.stats, stat.instructions))
            self.logger.debug('FlowDesc: %s', flows)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPFlowDescStatsReply, self).__init__(datapath, **kwargs)