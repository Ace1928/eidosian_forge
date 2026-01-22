import abc
import logging
import struct
import time
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from os_ken.lib.packet import bgp
from os_ken.lib.packet import ospf
class UnknownMrtMessage(MrtMessage):
    """
    MRT Message for the UNKNOWN Type.
    """

    def __init__(self, buf):
        self.buf = buf

    @classmethod
    def parse(cls, buf):
        return cls(buf)

    def serialize(self):
        return self.buf