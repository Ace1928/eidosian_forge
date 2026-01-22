import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
def _register_tlv_types(cls):
    operation._TLV_TYPES[type_] = cls
    return cls