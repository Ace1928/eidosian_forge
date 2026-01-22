import abc
import struct
from . import packet_base
from . import icmpv6
from . import tcp
from . import udp
from . import sctp
from . import gre
from . import in_proto as inet
from os_ken.lib import addrconv
from os_ken.lib import stringify
class opt_header(header):
    """an abstract class for Hop-by-Hop Options header and destination
    header."""
    _PACK_STR = '!BB'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _FIX_SIZE = 8
    _class_prefixes = ['option']

    @abc.abstractmethod
    def __init__(self, nxt, size, data):
        super(opt_header, self).__init__(nxt)
        assert not size % 8
        self.size = size
        self.data = data

    @classmethod
    def parser(cls, buf):
        nxt, len_ = struct.unpack_from(cls._PACK_STR, buf)
        data_len = cls._FIX_SIZE + int(len_)
        data = []
        size = cls._MIN_LEN
        while size < data_len:
            type_, = struct.unpack_from('!B', buf[size:])
            if type_ == 0:
                opt = option(type_, -1, None)
                size += 1
            else:
                opt = option.parser(buf[size:])
                size += len(opt)
            data.append(opt)
        return cls(nxt, len_, data)

    def serialize(self):
        buf = struct.pack(self._PACK_STR, self.nxt, self.size)
        buf = bytearray(buf)
        if self.data is None:
            self.data = [option(type_=1, len_=4, data=b'\x00\x00\x00\x00')]
        for opt in self.data:
            buf.extend(opt.serialize())
        return buf

    def __len__(self):
        return self._FIX_SIZE + self.size