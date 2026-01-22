from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
class LoadLocalPacketWrapper:
    """
    Load Local Packet Wrapper. It uses an existing packet object, and wraps
    around it, exposing useful variables while still providing access
    to the original packet objects variables and methods.
    """

    def __init__(self, from_packet):
        if not from_packet.is_load_local_packet():
            raise ValueError(f"Cannot create '{self.__class__}' object from invalid packet type")
        self.packet = from_packet
        self.filename = self.packet.get_all_data()[1:]
        if DEBUG:
            print('filename=', self.filename)

    def __getattr__(self, key):
        return getattr(self.packet, key)