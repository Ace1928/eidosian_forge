import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class chunk_init_base(chunk, metaclass=abc.ABCMeta):
    _PACK_STR = '!BBHIIHHI'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _class_prefixes = ['param_']

    def __init__(self, flags=0, length=0, init_tag=0, a_rwnd=0, os=0, mis=0, i_tsn=0, params=None):
        super(chunk_init_base, self).__init__(self.chunk_type(), length)
        self.flags = flags
        self.init_tag = init_tag
        self.a_rwnd = a_rwnd
        self.os = os
        self.mis = mis
        self.i_tsn = i_tsn
        params = params or []
        assert isinstance(params, list)
        for one in params:
            assert isinstance(one, param)
        self.params = params

    @classmethod
    def parser_base(cls, buf, recog):
        _, flags, length, init_tag, a_rwnd, os, mis, i_tsn = struct.unpack_from(cls._PACK_STR, buf)
        params = []
        offset = cls._MIN_LEN
        while offset < length:
            ptype, = struct.unpack_from('!H', buf, offset)
            cls_ = recog.get(ptype)
            if not cls_:
                break
            ins = cls_.parser(buf[offset:])
            params.append(ins)
            offset += len(ins)
        msg = cls(flags, length, init_tag, a_rwnd, os, mis, i_tsn, params)
        return msg

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.chunk_type(), self.flags, self.length, self.init_tag, self.a_rwnd, self.os, self.mis, self.i_tsn))
        for one in self.params:
            buf.extend(one.serialize())
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        return bytes(buf)