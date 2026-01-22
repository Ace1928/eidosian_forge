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
@OFPMultipartReply.register_stats_type()
@_set_stats_type(ofproto.OFPMP_QUEUE_DESC, OFPQueueDesc)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPQueueDescStatsReply(OFPMultipartReply):
    """
    Queue description reply message

    The switch responds with this message to a queue description request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPQueueDesc`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPQueueDescStatsReply, MAIN_DISPATCHER)
        def queue_desc_stats_reply_handler(self, ev):
            queues = []
            for q in ev.msg.body:
                queues.append('port_no=%d queue_id=0x%08x properties=%s' %
                             (q.port_no, q.queue_id, repr(q.properties)))
            self.logger.debug('OFPQueueDescStatsReply received: %s', queues)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPQueueDescStatsReply, self).__init__(datapath, **kwargs)