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
@Bgp4MpMrtMessage.register_type(Bgp4MpMrtRecord.SUBTYPE_BGP4MP_STATE_CHANGE)
class Bgp4MpStateChangeMrtMessage(Bgp4MpMrtMessage):
    """
    MRT Message for the BGP4MP Type and the BGP4MP_STATE_CHANGE subtype.
    """
    _HEADER_FMT = '!HHHH'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _ADDRS_FMT = '!%ds%ds'
    _STATES_FMT = '!HH'
    STATES_SIZE = struct.calcsize(_STATES_FMT)
    STATE_IDLE = 1
    STATE_CONNECT = 2
    STATE_ACTIVE = 3
    STATE_OPEN_SENT = 4
    STATE_OPEN_CONFIRM = 5
    STATE_ESTABLISHED = 6
    AFI_IPv4 = 1
    AFI_IPv6 = 2

    def __init__(self, peer_as, local_as, if_index, peer_ip, local_ip, old_state, new_state, afi=None):
        self.peer_as = peer_as
        self.local_as = local_as
        self.if_index = if_index
        self.afi = afi
        self.peer_ip = peer_ip
        self.local_ip = local_ip
        self.old_state = old_state
        self.new_state = new_state

    @classmethod
    def parse(cls, buf):
        peer_as, local_as, if_index, afi = struct.unpack_from(cls._HEADER_FMT, buf)
        offset = cls.HEADER_SIZE
        if afi == cls.AFI_IPv4:
            addrs_fmt = cls._ADDRS_FMT % (4, 4)
        elif afi == cls.AFI_IPv6:
            addrs_fmt = cls._ADDRS_FMT % (16, 16)
        else:
            raise struct.error('Unsupported address family: %d' % afi)
        peer_ip, local_ip = struct.unpack_from(addrs_fmt, buf, offset)
        peer_ip = ip.bin_to_text(peer_ip)
        local_ip = ip.bin_to_text(local_ip)
        offset += struct.calcsize(addrs_fmt)
        old_state, new_state = struct.unpack_from(cls._STATES_FMT, buf, offset)
        return cls(peer_as, local_as, if_index, peer_ip, local_ip, old_state, new_state, afi)

    def serialize(self):
        if ip.valid_ipv4(self.peer_ip) and ip.valid_ipv4(self.local_ip):
            self.afi = self.AFI_IPv4
        elif ip.valid_ipv6(self.peer_ip) and ip.valid_ipv6(self.local_ip):
            self.afi = self.AFI_IPv6
        else:
            raise ValueError('peer_ip and local_ip must be the same address family: peer_ip=%s, local_ip=%s' % (self.peer_ip, self.local_ip))
        buf = struct.pack(self._HEADER_FMT, self.peer_as, self.local_as, self.if_index, self.afi)
        buf += ip.text_to_bin(self.peer_ip)
        buf += ip.text_to_bin(self.local_ip)
        buf += struct.pack(self._STATES_FMT, self.old_state, self.new_state)
        return buf