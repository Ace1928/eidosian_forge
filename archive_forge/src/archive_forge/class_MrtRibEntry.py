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
class MrtRibEntry(stringify.StringifyMixin):
    """
    MRT RIB Entry.
    """
    _HEADER_FMT = '!HIH'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _HEADER_FMT_ADDPATH = '!HIIH'
    HEADER_SIZE_ADDPATH = struct.calcsize(_HEADER_FMT_ADDPATH)

    def __init__(self, peer_index, originated_time, bgp_attributes, attr_len=None, path_id=None):
        self.peer_index = peer_index
        self.originated_time = originated_time
        assert isinstance(bgp_attributes, (list, tuple))
        for attr in bgp_attributes:
            assert isinstance(attr, bgp._PathAttribute)
        self.bgp_attributes = bgp_attributes
        self.attr_len = attr_len
        self.path_id = path_id

    @classmethod
    def parse(cls, buf, is_addpath=False):
        path_id = None
        if not is_addpath:
            peer_index, originated_time, attr_len = struct.unpack_from(cls._HEADER_FMT, buf)
            _header_size = cls.HEADER_SIZE
        else:
            peer_index, originated_time, path_id, attr_len = struct.unpack_from(cls._HEADER_FMT_ADDPATH, buf)
            _header_size = cls.HEADER_SIZE_ADDPATH
        bgp_attr_bin = buf[_header_size:_header_size + attr_len]
        bgp_attributes = []
        while bgp_attr_bin:
            attr, bgp_attr_bin = bgp._PathAttribute.parser(bgp_attr_bin)
            bgp_attributes.append(attr)
        return (cls(peer_index, originated_time, bgp_attributes, attr_len, path_id), buf[_header_size + attr_len:])

    def serialize(self):
        bgp_attrs_bin = bytearray()
        for attr in self.bgp_attributes:
            bgp_attrs_bin += attr.serialize()
        self.attr_len = len(bgp_attrs_bin)
        if self.path_id is None:
            return struct.pack(self._HEADER_FMT, self.peer_index, self.originated_time, self.attr_len) + bgp_attrs_bin
        else:
            return struct.pack(self._HEADER_FMT_ADDPATH, self.peer_index, self.originated_time, self.path_id, self.attr_len) + bgp_attrs_bin