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
class OFPMeterBandHeader(OFPMeterBand):
    _METER_BAND = {}

    @staticmethod
    def register_meter_band_type(type_, len_):

        def _register_meter_band_type(cls):
            OFPMeterBandHeader._METER_BAND[type_] = cls
            cls.cls_meter_band_type = type_
            cls.cls_meter_band_len = len_
            return cls
        return _register_meter_band_type

    def __init__(self):
        cls = self.__class__
        super(OFPMeterBandHeader, self).__init__(cls.cls_meter_band_type, cls.cls_meter_band_len)

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, _rate, _burst_size = struct.unpack_from(ofproto.OFP_METER_BAND_HEADER_PACK_STR, buf, offset)
        cls_ = cls._METER_BAND[type_]
        assert cls_.cls_meter_band_len == len_
        return cls_.parser(buf, offset)