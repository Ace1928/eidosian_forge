from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@ExtendedPrefixTLV.register_type(OSPF_EXTENDED_PREFIX_SID_SUBTLV)
class PrefixSIDSubTLV(ExtendedPrefixTLV):
    _VALUE_PACK_STR = '!HHBBBBHHI'
    _VALUE_PACK_LEN = struct.calcsize(_VALUE_PACK_STR)
    _VALUE_FIELDS = ['flags', 'mt_id', 'algorithm', '_pad', 'range_size', '_pad', 'index']

    def __init__(self, type_=OSPF_EXTENDED_PREFIX_SID_SUBTLV, length=0, flags=0, mt_id=0, algorithm=0, range_size=0, index=0):
        super(PrefixSIDSubTLV, self).__init__()
        self.type_ = type_
        self.length = length
        self.flags = flags
        self.mt_id = mt_id
        self.algorithm = algorithm
        self.range_size = range_size
        self.index = index

    @classmethod
    def parser(cls, buf):
        rest = buf[cls._VALUE_PACK_LEN:]
        buf = buf[:cls._VALUE_PACK_LEN]
        type_, length, flags, mt_id, algorithm, _pad, range_size, _pad, index = struct.unpack_from(cls._VALUE_PACK_STR, buf)
        return (cls(type_, length, flags, mt_id, algorithm, range_size, index), rest)

    def serialize(self):
        return struct.pack(self._VALUE_PACK_STR, OSPF_EXTENDED_PREFIX_SID_SUBTLV, self._VALUE_PACK_LEN - 4, self.flags, self.mt_id, self.algorithm, 0, self.range_size, 0, self.index)