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
class OFPQueueProp(OFPQueuePropHeader):
    _QUEUE_PROP_PROPERTIES = {}

    @staticmethod
    def register_property(property_, len_=None):

        def _register_property(cls):
            cls.cls_property = property_
            cls.cls_len = len_
            OFPQueueProp._QUEUE_PROP_PROPERTIES[property_] = cls
            return cls
        return _register_property

    def __init__(self):
        cls = self.__class__
        super(OFPQueueProp, self).__init__(cls.cls_property, cls.cls_len)

    @classmethod
    def parser(cls, buf, offset):
        property_, len_ = struct.unpack_from(ofproto.OFP_QUEUE_PROP_HEADER_PACK_STR, buf, offset)
        cls_ = cls._QUEUE_PROP_PROPERTIES.get(property_)
        if cls_ is not None:
            p = cls_.parser(buf, offset + ofproto.OFP_QUEUE_PROP_HEADER_SIZE)
            p.property = property_
            p.len = len_
            if property_ == ofproto.OFPQT_EXPERIMENTER:
                rest = buf[offset + ofproto.OFP_QUEUE_PROP_EXPERIMENTER_SIZE:offset + len_]
                p.parse_experimenter_data(rest)
            return p
        return None