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
@OFPMultipartReply.register_stats_type(body_single_struct=True)
@_set_stats_type(ofproto.OFPMP_BUNDLE_FEATURES, OFPBundleFeaturesStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPBundleFeaturesStatsReply(OFPMultipartReply):
    """
    Bundle features reply message

    The switch responds with this message to a bundle features request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             Instance of ``OFPBundleFeaturesStats``
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPBundleFeaturesStatsReply, MAIN_DISPATCHER)
        def bundle_features_stats_reply_handler(self, ev):
            body = ev.msg.body

            self.logger.debug('OFPBundleFeaturesStats: capabilities=%0x%08x '
                              'properties=%s',
                              body.capabilities, repr(body.properties))
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPBundleFeaturesStatsReply, self).__init__(datapath, **kwargs)