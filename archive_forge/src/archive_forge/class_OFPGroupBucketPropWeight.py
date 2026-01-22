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
@OFPGroupBucketProp.register_type(ofproto.OFPGBPT_WEIGHT)
class OFPGroupBucketPropWeight(OFPGroupBucketProp):

    def __init__(self, type_=None, length=None, weight=None):
        super(OFPGroupBucketPropWeight, self).__init__(type_, length)
        self.weight = weight

    @classmethod
    def parser(cls, buf):
        prop = cls()
        prop.type, prop.length, prop.weight = struct.unpack_from(ofproto.OFP_GROUP_BUCKET_PROP_WEIGHT_PACK_STR, buf, 0)
        return prop

    def serialize(self):
        self.length = ofproto.OFP_GROUP_BUCKET_PROP_WEIGHT_SIZE
        buf = bytearray()
        msg_pack_into(ofproto.OFP_GROUP_BUCKET_PROP_WEIGHT_PACK_STR, buf, 0, self.type, self.length, self.weight)
        return buf