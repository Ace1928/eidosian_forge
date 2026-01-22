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
@OFPQueueProp.register_property(ofproto.OFPQT_EXPERIMENTER)
class OFPQueuePropExperimenter(OFPQueueProp):
    _EXPERIMENTER_DATA_PACK_STR = '!B'
    _EXPERIMENTER_DATA_SIZE = 1

    def __init__(self, experimenter, data=None, property_=None, len_=None):
        super(OFPQueuePropExperimenter, self).__init__()
        self.experimenter = experimenter
        self.data = data

    @classmethod
    def parser(cls, buf, offset):
        experimenter, = struct.unpack_from(ofproto.OFP_QUEUE_PROP_EXPERIMENTER_PACK_STR, buf, offset)
        return cls(experimenter)

    def parse_experimenter_data(self, rest):
        data = []
        while rest:
            d, = struct.unpack_from(self._EXPERIMENTER_DATA_PACK_STR, rest, 0)
            data.append(d)
            rest = rest[self._EXPERIMENTER_DATA_SIZE:]
        self.data = data