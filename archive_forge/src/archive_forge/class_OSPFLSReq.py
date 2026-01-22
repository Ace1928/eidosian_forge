from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@OSPFMessage.register_type(OSPF_MSG_LS_REQ)
class OSPFLSReq(OSPFMessage):
    _MIN_LEN = OSPFMessage._HDR_LEN

    class Request(StringifyMixin):
        _PACK_STR = '!I4s4s'
        _PACK_LEN = struct.calcsize(_PACK_STR)

        def __init__(self, type_=OSPF_UNKNOWN_LSA, id_='0.0.0.0', adv_router='0.0.0.0'):
            self.type_ = type_
            self.id = id_
            self.adv_router = adv_router

        @classmethod
        def parser(cls, buf):
            if len(buf) < cls._PACK_LEN:
                raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._PACK_LEN))
            link = buf[:cls._PACK_LEN]
            rest = buf[cls._PACK_LEN:]
            type_, id_, adv_router = struct.unpack_from(cls._PACK_STR, bytes(link))
            id_ = addrconv.ipv4.bin_to_text(id_)
            adv_router = addrconv.ipv4.bin_to_text(adv_router)
            return (cls(type_, id_, adv_router), rest)

        def serialize(self):
            id_ = addrconv.ipv4.text_to_bin(self.id)
            adv_router = addrconv.ipv4.text_to_bin(self.adv_router)
            return struct.pack(self._PACK_STR, self.type_, id_, adv_router)

    def __init__(self, length=None, router_id='0.0.0.0', area_id='0.0.0.0', au_type=1, authentication=0, checksum=None, version=_VERSION, lsa_requests=None):
        lsa_requests = lsa_requests if lsa_requests else []
        super(OSPFLSReq, self).__init__(OSPF_MSG_LS_REQ, length, router_id, area_id, au_type, authentication, checksum, version)
        self.lsa_requests = lsa_requests

    @classmethod
    def parser(cls, buf):
        reqs = []
        while buf:
            req, buf = cls.Request.parser(buf)
            reqs.append(req)
        return {'lsa_requests': reqs}

    def serialize_tail(self):
        return reduce(lambda a, b: a + b, (req.serialize() for req in self.lsa_requests))