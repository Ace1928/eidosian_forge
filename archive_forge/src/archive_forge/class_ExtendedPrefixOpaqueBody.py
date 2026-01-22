from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@OpaqueBody.register_type(OSPF_OPAQUE_TYPE_EXTENDED_PREFIX_LSA)
class ExtendedPrefixOpaqueBody(OpaqueBody):

    @classmethod
    def parser(cls, buf):
        buf = bytes(buf)
        tlvs = []
        while buf:
            type_, length = struct.unpack_from('!HH', buf)
            if len(buf[struct.calcsize('!HH'):]) < length:
                raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), length))
            tlvcls = ExtendedPrefixTLV._lookup_type(type_)
            if tlvcls:
                tlv, buf = tlvcls.parser(buf)
                tlvs.append(tlv)
        return cls(tlvs)