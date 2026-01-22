import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
@staticmethod
def _decode_port_id(port_id):
    priority = port_id >> 8 & 240
    port_number = port_id & 4095
    return (priority, port_number)