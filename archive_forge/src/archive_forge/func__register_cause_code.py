import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
def _register_cause_code(cls):
    chunk_error._RECOGNIZED_CAUSES[cls.cause_code()] = cls
    return cls