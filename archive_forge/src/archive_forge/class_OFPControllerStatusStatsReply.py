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
@_set_stats_type(ofproto.OFPMP_CONTROLLER_STATUS, OFPControllerStatusStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPControllerStatusStatsReply(OFPMultipartReply):
    """
     Controller status multipart reply message

    The switch responds with this message to a controller status
    multipart request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPControllerStatus`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPControllerStatusStatsReply,
                    MAIN_DISPATCHER)
        def controller_status_multipart_reply_handler(self, ev):
            status = []
            for s in ev.msg.body:
                status.append('short_id=%d role=%d reason=%d '
                              'channel_status=%d properties=%s' %
                              (s.short_id, s.role, s.reason,
                               s.channel_status, repr(s.properties)))
            self.logger.debug('OFPControllerStatusStatsReply received: %s',
                              status)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPControllerStatusStatsReply, self).__init__(datapath, **kwargs)