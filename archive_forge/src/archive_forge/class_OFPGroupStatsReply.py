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
@OFPMultipartReply.register_stats_type()
@_set_stats_type(ofproto.OFPMP_GROUP, OFPGroupStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPGroupStatsReply(OFPMultipartReply):
    """
    Group statistics reply message

    The switch responds with this message to a group statistics request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPGroupStats`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPGroupStatsReply, MAIN_DISPATCHER)
        def group_stats_reply_handler(self, ev):
            groups = []
            for stat in ev.msg.body:
                groups.append('length=%d group_id=%d '
                              'ref_count=%d packet_count=%d byte_count=%d '
                              'duration_sec=%d duration_nsec=%d' %
                              (stat.length, stat.group_id,
                               stat.ref_count, stat.packet_count,
                               stat.byte_count, stat.duration_sec,
                               stat.duration_nsec))
            self.logger.debug('GroupStats: %s', groups)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPGroupStatsReply, self).__init__(datapath, **kwargs)