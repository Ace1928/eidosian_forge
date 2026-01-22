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
class OFPBundleFeaturesStats(ofproto_parser.namedtuple('OFPBundleFeaturesStats', ('capabilities', 'properties'))):

    @classmethod
    def parser(cls, buf, offset):
        capabilities, = struct.unpack_from(ofproto.OFP_BUNDLE_FEATURES_PACK_STR, buf, offset)
        properties = []
        length = ofproto.OFP_BUNDLE_FEATURES_SIZE
        rest = buf[offset + length:]
        while rest:
            p, rest = OFPBundleFeaturesProp.parse(rest)
            properties.append(p)
            length += p.length
        bndl = cls(capabilities, properties)
        bndl.length = length
        return bndl