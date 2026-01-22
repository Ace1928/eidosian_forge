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
@_set_stats_type(ofproto.OFPMP_FLOW_MONITOR, OFPFlowUpdateHeader)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPFlowMonitorReply(OFPMultipartReply):
    """
    Flow monitor reply message

    The switch responds with this message to a flow monitor request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of list of the following class instance.

                     | OFPFlowMonitorFull
                     | OFPFlowMonitorAbbrev
                     | OFPFlowMonitorPaused
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPFlowMonitorReply, MAIN_DISPATCHER)
        def flow_monitor_reply_handler(self, ev):
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto
            flow_updates = []

            for update in msg.body:
                update_str = 'length=%d event=%d' %
                             (update.length, update.event)
                if (update.event == ofp.OFPFME_INITIAL or
                    update.event == ofp.OFPFME_ADDED or
                    update.event == ofp.OFPFME_REMOVED or
                    update.event == ofp.OFPFME_MODIFIED):
                    update_str += 'table_id=%d reason=%d idle_timeout=%d '
                                  'hard_timeout=%d priority=%d cookie=%d '
                                  'match=%d instructions=%s' %
                                  (update.table_id, update.reason,
                                   update.idle_timeout, update.hard_timeout,
                                   update.priority, update.cookie,
                                   update.match, update.instructions)
                elif update.event == ofp.OFPFME_ABBREV:
                    update_str += 'xid=%d' % (update.xid)
                flow_updates.append(update_str)
            self.logger.debug('FlowUpdates: %s', flow_updates)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPFlowMonitorReply, self).__init__(datapath, **kwargs)