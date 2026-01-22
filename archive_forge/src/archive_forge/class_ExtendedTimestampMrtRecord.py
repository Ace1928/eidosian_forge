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
class ExtendedTimestampMrtRecord(MrtRecord):
    """
    MRT record using Extended Timestamp MRT Header.
    """
    _HEADER_FMT = '!IHHII'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _EXT_HEADER_FMT = '!I'
    EXT_HEADER_SIZE = struct.calcsize(_EXT_HEADER_FMT)

    def __init__(self, message, timestamp=None, type_=None, subtype=None, ms_timestamp=None, length=None):
        super(ExtendedTimestampMrtRecord, self).__init__(message, timestamp, type_, subtype, length)
        self.ms_timestamp = ms_timestamp

    @classmethod
    def parse_extended_header(cls, buf):
        ms_timestamp, = struct.unpack_from(cls._EXT_HEADER_FMT, buf)
        return ([ms_timestamp], buf[cls.EXT_HEADER_SIZE:])

    def serialize_header(self):
        return struct.pack(self._HEADER_FMT, self.timestamp, self.type, self.subtype, self.length, self.ms_timestamp)