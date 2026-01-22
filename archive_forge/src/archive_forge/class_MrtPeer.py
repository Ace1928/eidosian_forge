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
class MrtPeer(stringify.StringifyMixin):
    """
    MRT Peer.
    """
    _HEADER_FMT = '!B4s'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    IP_ADDR_FAMILY_BIT = 1 << 0
    AS_NUMBER_SIZE_BIT = 1 << 1
    _TYPE = {'ascii': ['bgp_id', 'ip_addr']}

    def __init__(self, bgp_id, ip_addr, as_num, type_=0):
        self.type = type_
        self.bgp_id = bgp_id
        self.ip_addr = ip_addr
        self.as_num = as_num

    @classmethod
    def parse(cls, buf):
        type_, bgp_id = struct.unpack_from(cls._HEADER_FMT, buf)
        bgp_id = addrconv.ipv4.bin_to_text(bgp_id)
        offset = cls.HEADER_SIZE
        if type_ & cls.IP_ADDR_FAMILY_BIT:
            ip_addr_len = 16
        else:
            ip_addr_len = 4
        ip_addr = ip.bin_to_text(buf[offset:offset + ip_addr_len])
        offset += ip_addr_len
        if type_ & cls.AS_NUMBER_SIZE_BIT:
            as_num, = struct.unpack_from('!I', buf, offset)
            offset += 4
        else:
            as_num, = struct.unpack_from('!H', buf, offset)
            offset += 2
        return (cls(bgp_id, ip_addr, as_num, type_), buf[offset:])

    def serialize(self):
        if ip.valid_ipv6(self.ip_addr):
            self.type |= self.IP_ADDR_FAMILY_BIT
        ip_addr = ip.text_to_bin(self.ip_addr)
        if self.type & self.AS_NUMBER_SIZE_BIT or self.as_num > 65535:
            self.type |= self.AS_NUMBER_SIZE_BIT
            as_num = struct.pack('!I', self.as_num)
        else:
            as_num = struct.pack('!H', self.as_num)
        buf = struct.pack(self._HEADER_FMT, self.type, addrconv.ipv4.text_to_bin(self.bgp_id))
        return buf + ip_addr + as_num