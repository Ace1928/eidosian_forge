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
class OFPMeterStats(StringifyMixin):

    def __init__(self, meter_id=None, flow_count=None, packet_in_count=None, byte_in_count=None, duration_sec=None, duration_nsec=None, band_stats=None, len_=None):
        super(OFPMeterStats, self).__init__()
        self.meter_id = meter_id
        self.len = 0
        self.flow_count = flow_count
        self.packet_in_count = packet_in_count
        self.byte_in_count = byte_in_count
        self.duration_sec = duration_sec
        self.duration_nsec = duration_nsec
        self.band_stats = band_stats

    @classmethod
    def parser(cls, buf, offset):
        meter_stats = cls()
        meter_stats.meter_id, meter_stats.len, meter_stats.flow_count, meter_stats.packet_in_count, meter_stats.byte_in_count, meter_stats.duration_sec, meter_stats.duration_nsec = struct.unpack_from(ofproto.OFP_METER_STATS_PACK_STR, buf, offset)
        offset += ofproto.OFP_METER_STATS_SIZE
        meter_stats.band_stats = []
        length = ofproto.OFP_METER_STATS_SIZE
        while length < meter_stats.len:
            band_stats = OFPMeterBandStats.parser(buf, offset)
            meter_stats.band_stats.append(band_stats)
            offset += ofproto.OFP_METER_BAND_STATS_SIZE
            length += ofproto.OFP_METER_BAND_STATS_SIZE
        return meter_stats