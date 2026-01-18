import struct
import logging
from . import packet_base
from os_ken.lib import type_desc
def vni_from_bin(buf):
    """
    Converts binary representation VNI to integer.

    :param buf: binary representation of VNI.
    :return: VNI integer.
    """
    return type_desc.Int3.to_user(bytes(buf))