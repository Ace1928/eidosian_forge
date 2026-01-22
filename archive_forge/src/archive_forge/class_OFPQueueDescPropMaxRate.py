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
@OFPQueueDescProp.register_type(ofproto.OFPQDPT_MAX_RATE)
class OFPQueueDescPropMaxRate(OFPQueueDescProp):

    def __init__(self, type_=None, length=None, rate=None):
        self.type = type_
        self.length = length
        self.rate = rate

    @classmethod
    def parser(cls, buf):
        maxrate = cls()
        maxrate.type, maxrate.length, maxrate.rate = struct.unpack_from(ofproto.OFP_QUEUE_DESC_PROP_MAX_RATE_PACK_STR, buf, 0)
        return maxrate