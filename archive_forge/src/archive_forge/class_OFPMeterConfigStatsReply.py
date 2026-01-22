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
@_set_stats_type(ofproto.OFPMP_METER_CONFIG, OFPMeterConfigStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPMeterConfigStatsReply(OFPMultipartReply):
    """
    Meter configuration statistics reply message

    The switch responds with this message to a meter configuration
    statistics request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPMeterConfigStats`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPMeterConfigStatsReply, MAIN_DISPATCHER)
        def meter_config_stats_reply_handler(self, ev):
            configs = []
            for stat in ev.msg.body:
                configs.append('length=%d flags=0x%04x meter_id=0x%08x '
                               'bands=%s' %
                               (stat.length, stat.flags, stat.meter_id,
                                stat.bands))
            self.logger.debug('MeterConfigStats: %s', configs)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPMeterConfigStatsReply, self).__init__(datapath, **kwargs)