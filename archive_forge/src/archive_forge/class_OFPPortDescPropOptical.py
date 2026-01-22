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
@OFPPortDescProp.register_type(ofproto.OFPPDPT_OPTICAL)
class OFPPortDescPropOptical(OFPPortDescProp):

    def __init__(self, type_=None, length=None, supported=None, tx_min_freq_lmda=None, tx_max_freq_lmda=None, tx_grid_freq_lmda=None, rx_min_freq_lmda=None, rx_max_freq_lmda=None, rx_grid_freq_lmda=None, tx_pwr_min=None, tx_pwr_max=None):
        self.type = type_
        self.length = length
        self.supported = supported
        self.tx_min_freq_lmda = tx_min_freq_lmda
        self.tx_max_freq_lmda = tx_max_freq_lmda
        self.tx_grid_freq_lmda = tx_grid_freq_lmda
        self.rx_min_freq_lmda = rx_min_freq_lmda
        self.rx_max_freq_lmda = rx_max_freq_lmda
        self.rx_grid_freq_lmda = rx_grid_freq_lmda
        self.tx_pwr_min = tx_pwr_min
        self.tx_pwr_max = tx_pwr_max

    @classmethod
    def parser(cls, buf):
        optical = cls()
        optical.type, optical.length, optical.supported, optical.tx_min_freq_lmda, optical.tx_max_freq_lmda, optical.tx_grid_freq_lmda, optical.rx_min_freq_lmda, optical.rx_max_freq_lmda, optical.rx_grid_freq_lmda, optical.tx_pwr_min, optical.tx_pwr_max = struct.unpack_from(ofproto.OFP_PORT_DESC_PROP_OPTICAL_PACK_STR, buf, 0)
        return optical