from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@OSPFMessage.register_type(OSPF_MSG_LS_UPD)
class OSPFLSUpd(OSPFMessage):
    _PACK_STR = '!I'
    _PACK_LEN = struct.calcsize(_PACK_STR)
    _MIN_LEN = OSPFMessage._HDR_LEN + _PACK_LEN

    def __init__(self, length=None, router_id='0.0.0.0', area_id='0.0.0.0', au_type=1, authentication=0, checksum=None, version=_VERSION, lsas=None):
        lsas = lsas if lsas else []
        super(OSPFLSUpd, self).__init__(OSPF_MSG_LS_UPD, length, router_id, area_id, au_type, authentication, checksum, version)
        self.lsas = lsas

    @classmethod
    def parser(cls, buf):
        binnum = buf[:cls._PACK_LEN]
        num, = struct.unpack_from(cls._PACK_STR, bytes(binnum))
        buf = buf[cls._PACK_LEN:]
        lsas = []
        while buf:
            lsa, _cls, buf = LSA.parser(buf)
            lsas.append(lsa)
        assert len(lsas) == num
        return {'lsas': lsas}

    def serialize_tail(self):
        head = bytearray(struct.pack(self._PACK_STR, len(self.lsas)))
        try:
            return head + reduce(lambda a, b: a + b, (lsa.serialize() for lsa in self.lsas))
        except TypeError:
            return head