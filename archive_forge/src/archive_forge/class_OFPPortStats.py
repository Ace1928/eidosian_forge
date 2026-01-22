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
class OFPPortStats(ofproto_parser.namedtuple('OFPPortStats', ('port_no', 'rx_packets', 'tx_packets', 'rx_bytes', 'tx_bytes', 'rx_dropped', 'tx_dropped', 'rx_errors', 'tx_errors', 'rx_frame_err', 'rx_over_err', 'rx_crc_err', 'collisions', 'duration_sec', 'duration_nsec'))):

    @classmethod
    def parser(cls, buf, offset):
        port = struct.unpack_from(ofproto.OFP_PORT_STATS_PACK_STR, buf, offset)
        stats = cls(*port)
        stats.length = ofproto.OFP_PORT_STATS_SIZE
        return stats