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
@OFPPortStatsProp.register_type(ofproto.OFPPSPT_ETHERNET)
class OFPPortStatsPropEthernet(OFPPortStatsProp):

    def __init__(self, type_=None, length=None, rx_frame_err=None, rx_over_err=None, rx_crc_err=None, collisions=None):
        self.type = type_
        self.length = length
        self.rx_frame_err = rx_frame_err
        self.rx_over_err = rx_over_err
        self.rx_crc_err = rx_crc_err
        self.collisions = collisions

    @classmethod
    def parser(cls, buf):
        ether = cls()
        ether.type, ether.length, ether.rx_frame_err, ether.rx_over_err, ether.rx_crc_err, ether.collisions = struct.unpack_from(ofproto.OFP_PORT_STATS_PROP_ETHERNET_PACK_STR, buf, 0)
        return ether