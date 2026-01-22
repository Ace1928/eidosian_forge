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
class OFPActionId(StringifyMixin):
    _PACK_STR = '!HH'

    def __init__(self, type_, len_=None):
        self.type = type_
        self.len = len_

    @classmethod
    def parse(cls, buf):
        type_, len_ = struct.unpack_from(cls._PACK_STR, bytes(buf), 0)
        rest = buf[len_:]
        return (cls(type_=type_, len_=len_), rest)

    def serialize(self):
        self.len = struct.calcsize(self._PACK_STR)
        buf = bytearray()
        msg_pack_into(self._PACK_STR, buf, 0, self.type, self.len)
        return buf