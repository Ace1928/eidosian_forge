import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
class MFIPV6(object):
    pack_str = MF_PACK_STRING_IPV6

    @classmethod
    def field_parser(cls, header, buf, offset):
        hasmask = header >> 8 & 1
        if hasmask:
            pack_string = '!' + cls.pack_str[1:] * 2
            value = struct.unpack_from(pack_string, buf, offset + 4)
            return cls(header, list(value[:8]), list(value[8:]))
        else:
            value = struct.unpack_from(cls.pack_str, buf, offset + 4)
            return cls(header, list(value))