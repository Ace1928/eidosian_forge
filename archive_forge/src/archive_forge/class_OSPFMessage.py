from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
class OSPFMessage(packet_base.PacketBase, type_desc.TypeDisp):
    """Base class for OSPF version 2 messages.
    """
    _HDR_PACK_STR = '!BBH4s4sHHQ'
    _HDR_LEN = struct.calcsize(_HDR_PACK_STR)

    def __init__(self, type_, length=None, router_id='0.0.0.0', area_id='0.0.0.0', au_type=1, authentication=0, checksum=None, version=_VERSION):
        super(OSPFMessage, self).__init__()
        self.version = version
        self.type_ = type_
        self.length = length
        self.router_id = router_id
        self.area_id = area_id
        self.checksum = checksum
        self.au_type = au_type
        self.authentication = authentication

    @classmethod
    def _parser(cls, buf):
        if len(buf) < cls._HDR_LEN:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._HDR_LEN))
        version, type_, length, router_id, area_id, checksum, au_type, authentication = struct.unpack_from(cls._HDR_PACK_STR, bytes(buf))
        if packet_utils.checksum(buf[:12] + buf[14:16] + buf[cls._HDR_LEN:]) != checksum:
            raise InvalidChecksum
        if len(buf) < length:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), length))
        router_id = addrconv.ipv4.bin_to_text(router_id)
        area_id = addrconv.ipv4.bin_to_text(area_id)
        binmsg = buf[cls._HDR_LEN:length]
        rest = buf[length:]
        subcls = cls._lookup_type(type_)
        kwargs = subcls.parser(binmsg)
        return (subcls(length, router_id, area_id, au_type, int(authentication), checksum, version, **kwargs), None, rest)

    @classmethod
    def parser(cls, buf):
        try:
            return cls._parser(buf)
        except:
            return (None, None, buf)

    def serialize(self, payload=None, prev=None):
        tail = self.serialize_tail()
        self.length = self._HDR_LEN + len(tail)
        head = bytearray(struct.pack(self._HDR_PACK_STR, self.version, self.type_, self.length, addrconv.ipv4.text_to_bin(self.router_id), addrconv.ipv4.text_to_bin(self.area_id), 0, self.au_type, self.authentication))
        buf = head + tail
        csum = packet_utils.checksum(buf[:12] + buf[14:16] + buf[self._HDR_LEN:])
        self.checksum = csum
        struct.pack_into('!H', buf, 12, csum)
        return buf