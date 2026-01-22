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
class OFPMeterDescStats(StringifyMixin):

    def __init__(self, flags=None, meter_id=None, bands=None, length=None):
        super(OFPMeterDescStats, self).__init__()
        self.length = None
        self.flags = flags
        self.meter_id = meter_id
        self.bands = bands

    @classmethod
    def parser(cls, buf, offset):
        meter_config = cls()
        meter_config.length, meter_config.flags, meter_config.meter_id = struct.unpack_from(ofproto.OFP_METER_DESC_PACK_STR, buf, offset)
        offset += ofproto.OFP_METER_DESC_SIZE
        meter_config.bands = []
        length = ofproto.OFP_METER_DESC_SIZE
        while length < meter_config.length:
            band = OFPMeterBandHeader.parser(buf, offset)
            meter_config.bands.append(band)
            offset += band.len
            length += band.len
        return meter_config