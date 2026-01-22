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
class OFPDescStats(ofproto_parser.namedtuple('OFPDescStats', ('mfr_desc', 'hw_desc', 'sw_desc', 'serial_num', 'dp_desc'))):
    _TYPE = {'ascii': ['mfr_desc', 'hw_desc', 'sw_desc', 'serial_num', 'dp_desc']}

    @classmethod
    def parser(cls, buf, offset):
        desc = struct.unpack_from(ofproto.OFP_DESC_PACK_STR, buf, offset)
        desc = list(desc)
        desc = [x.rstrip(b'\x00') for x in desc]
        stats = cls(*desc)
        stats.length = ofproto.OFP_DESC_SIZE
        return stats