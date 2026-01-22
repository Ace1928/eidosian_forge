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
@TableDump2MrtMessage.register_type(TableDump2MrtRecord.SUBTYPE_PEER_INDEX_TABLE)
class TableDump2PeerIndexTableMrtMessage(TableDump2MrtMessage):
    """
    MRT Message for the TABLE_DUMP_V2 Type and the PEER_INDEX_TABLE subtype.
    """
    _HEADER_FMT = '!4sH'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _PEER_COUNT_FMT = '!H'
    PEER_COUNT_SIZE = struct.calcsize(_PEER_COUNT_FMT)
    _TYPE = {'ascii': ['bgp_id']}

    def __init__(self, bgp_id, peer_entries, view_name='', view_name_len=None, peer_count=None):
        self.bgp_id = bgp_id
        assert isinstance(peer_entries, (list, tuple))
        for p in peer_entries:
            assert isinstance(p, MrtPeer)
        self.peer_entries = peer_entries
        assert isinstance(view_name, str)
        self.view_name = view_name
        self.view_name_len = view_name_len
        self.peer_count = peer_count

    @classmethod
    def parse(cls, buf):
        bgp_id, view_name_len = struct.unpack_from(cls._HEADER_FMT, buf)
        bgp_id = addrconv.ipv4.bin_to_text(bgp_id)
        offset = cls.HEADER_SIZE
        view_name, = struct.unpack_from('!%ds' % view_name_len, buf, offset)
        view_name = str(view_name.decode('utf-8'))
        offset += view_name_len
        peer_count, = struct.unpack_from(cls._PEER_COUNT_FMT, buf, offset)
        offset += cls.PEER_COUNT_SIZE
        rest = buf[offset:]
        peer_entries = []
        for i in range(peer_count):
            p, rest = MrtPeer.parse(rest)
            peer_entries.insert(i, p)
        return cls(bgp_id, peer_entries, view_name, view_name_len, peer_count)

    def serialize(self):
        view_name = self.view_name.encode('utf-8')
        self.view_name_len = len(view_name)
        self.peer_count = len(self.peer_entries)
        buf = struct.pack(self._HEADER_FMT, addrconv.ipv4.text_to_bin(self.bgp_id), self.view_name_len) + view_name
        buf += struct.pack(self._PEER_COUNT_FMT, self.peer_count)
        for p in self.peer_entries:
            buf += p.serialize()
        return buf