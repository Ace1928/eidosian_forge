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
@OFPControllerStatusProp.register_type(ofproto.OFPCSPT_URI)
class OFPControllerStatusPropUri(OFPControllerStatusProp):
    _TYPE = {'ascii': ['uri']}

    def __init__(self, type_=None, length=None, uri=None):
        super(OFPControllerStatusPropUri, self).__init__(type_, length)
        self.uri = uri

    @classmethod
    def parser(cls, buf):
        rest = cls.get_rest(buf)
        pack_str = '!%ds' % len(rest)
        uri, = struct.unpack_from(pack_str, rest, 0)
        return cls(uri=uri)