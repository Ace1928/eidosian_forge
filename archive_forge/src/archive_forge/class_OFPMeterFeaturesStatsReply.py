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
@_set_stats_type(ofproto.OFPMP_METER_FEATURES, OFPMeterFeaturesStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPMeterFeaturesStatsReply(OFPMultipartReply):
    """
    Meter features statistics reply message

    The switch responds with this message to a meter features statistics
    request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPMeterFeaturesStats`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPMeterFeaturesStatsReply, MAIN_DISPATCHER)
        def meter_features_stats_reply_handler(self, ev):
            features = []
            for stat in ev.msg.body:
                features.append('max_meter=%d band_types=0x%08x '
                                'capabilities=0x%08x max_bands=%d '
                                'max_color=%d' %
                                (stat.max_meter, stat.band_types,
                                 stat.capabilities, stat.max_bands,
                                 stat.max_color))
            self.logger.debug('MeterFeaturesStats: %s', features)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPMeterFeaturesStatsReply, self).__init__(datapath, **kwargs)