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
@OFPPortDescProp.register_type(ofproto.OFPPDPT_RECIRCULATE)
class OFPPortDescPropRecirculate(OFPPortDescProp):
    _PORT_NO_PACK_STR = '!I'

    def __init__(self, type_=None, length=None, port_nos=None):
        port_nos = port_nos if port_nos else []
        super(OFPPortDescPropRecirculate, self).__init__(type_, length)
        self.port_nos = port_nos

    @classmethod
    def parser(cls, buf):
        rest = cls.get_rest(buf)
        nos = []
        while rest:
            n, = struct.unpack_from(cls._PORT_NO_PACK_STR, bytes(rest), 0)
            rest = rest[struct.calcsize(cls._PORT_NO_PACK_STR):]
            nos.append(n)
        return cls(port_nos=nos)

    def serialize_body(self):
        bin_nos = bytearray()
        for n in self.port_nos:
            bin_no = bytearray()
            msg_pack_into(self._PORT_NO_PACK_STR, bin_no, 0, n)
            bin_nos += bin_no
        return bin_nos