import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
class nd_option_la(nd_option):
    _PACK_STR = '!BB6s'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': ['hw_src']}

    @abc.abstractmethod
    def __init__(self, length, hw_src, data):
        super(nd_option_la, self).__init__(self.option_type(), length)
        self.hw_src = hw_src
        self.data = data

    @classmethod
    def parser(cls, buf, offset):
        _, length, hw_src = struct.unpack_from(cls._PACK_STR, buf, offset)
        msg = cls(length, addrconv.mac.bin_to_text(hw_src))
        offset += cls._MIN_LEN
        if len(buf) > offset:
            msg.data = buf[offset:]
        return msg

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.option_type(), self.length, addrconv.mac.text_to_bin(self.hw_src)))
        if self.data is not None:
            buf.extend(self.data)
        mod = len(buf) % 8
        if mod:
            buf.extend(bytearray(8 - mod))
        if 0 == self.length:
            self.length = len(buf) // 8
            struct.pack_into('!B', buf, 1, self.length)
        return bytes(buf)

    def __len__(self):
        length = self._MIN_LEN
        if self.data is not None:
            length += len(self.data)
        return length