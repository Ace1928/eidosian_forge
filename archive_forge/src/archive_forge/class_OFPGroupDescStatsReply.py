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
@_set_stats_type(ofproto.OFPMP_GROUP_DESC, OFPGroupDescStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPGroupDescStatsReply(OFPMultipartReply):
    """
    Group description reply message

    The switch responds with this message to a group description request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPGroupDescStats`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPGroupDescStatsReply, MAIN_DISPATCHER)
        def group_desc_stats_reply_handler(self, ev):
            descs = []
            for stat in ev.msg.body:
                descs.append('length=%d type=%d group_id=%d '
                             'buckets=%s' %
                             (stat.length, stat.type, stat.group_id,
                              stat.bucket))
            self.logger.debug('GroupDescStats: %s', descs)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPGroupDescStatsReply, self).__init__(datapath, **kwargs)