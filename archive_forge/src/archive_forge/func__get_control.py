import struct
from . import bpdu
from . import packet_base
from os_ken.lib import stringify
@classmethod
def _get_control(cls, buf):
    key = buf & 1 if buf & 1 == ControlFormatI.TYPE else buf & 3
    return cls._CTR_TYPES[key]