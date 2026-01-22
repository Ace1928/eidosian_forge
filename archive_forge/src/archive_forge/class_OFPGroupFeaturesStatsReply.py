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
@OFPMultipartReply.register_stats_type(body_single_struct=True)
@_set_stats_type(ofproto.OFPMP_GROUP_FEATURES, OFPGroupFeaturesStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPGroupFeaturesStatsReply(OFPMultipartReply):
    """
    Group features reply message

    The switch responds with this message to a group features request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             Instance of ``OFPGroupFeaturesStats``
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPGroupFeaturesStatsReply, MAIN_DISPATCHER)
        def group_features_stats_reply_handler(self, ev):
            body = ev.msg.body

            self.logger.debug('GroupFeaturesStats: types=%d '
                              'capabilities=0x%08x max_groups=%s '
                              'actions=%s',
                              body.types, body.capabilities,
                              body.max_groups, body.actions)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPGroupFeaturesStatsReply, self).__init__(datapath, **kwargs)