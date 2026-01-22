import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@staticmethod
def _serialize_tlvs(tlvs):
    buf = bytearray()
    for tlv_ in tlvs:
        buf.extend(tlv_.serialize())
    return buf