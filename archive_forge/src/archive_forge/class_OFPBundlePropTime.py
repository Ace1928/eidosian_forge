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
@OFPBundleProp.register_type(ofproto.OFPBPT_TIME)
class OFPBundlePropTime(OFPBundleProp):

    def __init__(self, type_=None, length=None, scheduled_time=None):
        super(OFPBundlePropTime, self).__init__(type_, length)
        self.scheduled_time = scheduled_time

    @classmethod
    def parser(cls, buf):
        prop = cls()
        offset = ofproto.OFP_BUNDLE_PROP_TIME_PACK_STR0_SIZE
        prop.scheduled_time = OFPTime.parser(buf, offset)
        return prop

    def serialize(self):
        self.length = ofproto.OFP_BUNDLE_PROP_TIME_PACK_STR_SIZE
        buf = bytearray()
        msg_pack_into(ofproto.OFP_BUNDLE_PROP_TIME_PACK_STR0, buf, 0, self.type, self.length)
        offset = ofproto.OFP_BUNDLE_PROP_TIME_PACK_STR0_SIZE
        self.scheduled_time.serialize(buf, offset)
        return buf