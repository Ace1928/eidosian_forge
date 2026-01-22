import inspect
import struct
import base64
from . import packet_base
from . import ethernet
from os_ken import utils
from os_ken.lib.stringify import StringifyMixin
def add_protocol(self, proto):
    """Register a protocol *proto* for this packet.

        This method is legal only when encoding a packet.

        When encoding a packet, register a protocol (ethernet, ipv4, ...)
        header to add to this packet.
        Protocol headers should be registered in on-wire order before calling
        self.serialize.
        """
    self.protocols.append(proto)