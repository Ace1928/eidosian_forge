import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@classmethod
def _parser_tlvs(cls, buf):
    offset = 0
    tlvs = []
    while True:
        type_, = struct.unpack_from('!B', buf, offset)
        cls_ = cls._TLV_TYPES.get(type_)
        if not cls_:
            assert type_ is CFM_END_TLV
            break
        tlv_ = cls_.parser(buf[offset:])
        tlvs.append(tlv_)
        offset += len(tlv_)
    return tlvs