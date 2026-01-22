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
@OFPPortStatsProp.register_type(ofproto.OFPPSPT_OPTICAL)
class OFPPortStatsPropOptical(OFPPortStatsProp):

    def __init__(self, type_=None, length=None, flags=None, tx_freq_lmda=None, tx_offset=None, tx_grid_span=None, rx_freq_lmda=None, rx_offset=None, rx_grid_span=None, tx_pwr=None, rx_pwr=None, bias_current=None, temperature=None):
        self.type = type_
        self.length = length
        self.flags = flags
        self.tx_freq_lmda = tx_freq_lmda
        self.tx_offset = tx_offset
        self.tx_grid_span = tx_grid_span
        self.rx_freq_lmda = rx_freq_lmda
        self.rx_offset = rx_offset
        self.rx_grid_span = rx_grid_span
        self.tx_pwr = tx_pwr
        self.rx_pwr = rx_pwr
        self.bias_current = bias_current
        self.temperature = temperature

    @classmethod
    def parser(cls, buf):
        optical = cls()
        optical.type, optical.length, optical.flags, optical.tx_freq_lmda, optical.tx_offset, optical.tx_grid_span, optical.rx_freq_lmda, optical.rx_offset, optical.rx_grid_span, optical.tx_pwr, optical.rx_pwr, optical.bias_current, optical.temperature = struct.unpack_from(ofproto.OFP_PORT_STATS_PROP_OPTICAL_PACK_STR, buf, 0)
        return optical