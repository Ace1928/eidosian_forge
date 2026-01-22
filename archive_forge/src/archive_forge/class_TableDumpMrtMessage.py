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
class TableDumpMrtMessage(MrtMessage, metaclass=abc.ABCMeta):
    """
    MRT Message for the TABLE_DUMP Type.
    """
    _HEADER_FMT = ''
    HEADER_SIZE = 0
    _TYPE = {'ascii': ['prefix', 'peer_ip']}

    def __init__(self, view_num, seq_num, prefix, prefix_len, status, originated_time, peer_ip, peer_as, bgp_attributes, attr_len=None):
        self.view_num = view_num
        self.seq_num = seq_num
        self.prefix = prefix
        self.prefix_len = prefix_len
        assert status == 1
        self.status = status
        self.originated_time = originated_time
        self.peer_ip = peer_ip
        self.peer_as = peer_as
        self.attr_len = attr_len
        assert isinstance(bgp_attributes, (list, tuple))
        for attr in bgp_attributes:
            assert isinstance(attr, bgp._PathAttribute)
        self.bgp_attributes = bgp_attributes

    @classmethod
    def parse(cls, buf):
        view_num, seq_num, prefix, prefix_len, status, originated_time, peer_ip, peer_as, attr_len = struct.unpack_from(cls._HEADER_FMT, buf)
        prefix = ip.bin_to_text(prefix)
        peer_ip = ip.bin_to_text(peer_ip)
        bgp_attr_bin = buf[cls.HEADER_SIZE:cls.HEADER_SIZE + attr_len]
        bgp_attributes = []
        while bgp_attr_bin:
            attr, bgp_attr_bin = bgp._PathAttribute.parser(bgp_attr_bin)
            bgp_attributes.append(attr)
        return cls(view_num, seq_num, prefix, prefix_len, status, originated_time, peer_ip, peer_as, bgp_attributes, attr_len)

    def serialize(self):
        bgp_attrs_bin = bytearray()
        for attr in self.bgp_attributes:
            bgp_attrs_bin += attr.serialize()
        self.attr_len = len(bgp_attrs_bin)
        prefix = ip.text_to_bin(self.prefix)
        peer_ip = ip.text_to_bin(self.peer_ip)
        return struct.pack(self._HEADER_FMT, self.view_num, self.seq_num, prefix, self.prefix_len, self.status, self.originated_time, peer_ip, self.peer_as, self.attr_len) + bgp_attrs_bin