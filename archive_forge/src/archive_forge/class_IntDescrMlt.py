import base64
import struct
from os_ken.lib import addrconv
class IntDescrMlt(TypeDescr):

    def __init__(self, length, num):
        self.length = length
        self.num = num
        self.size = length * num

    def to_user(self, binary):
        assert len(binary) == self.size
        lb = _split_str(binary, self.length)
        li = []
        for b in lb:
            i = 0
            for _ in range(self.length):
                c = b[:1]
                i = i * 256 + ord(c)
                b = b[1:]
            li.append(i)
        return tuple(li)

    def from_user(self, li):
        assert len(li) == self.num
        binary = b''
        for i in li:
            b = b''
            for _ in range(self.length):
                b = struct.Struct('>B').pack(i & 255) + b
                i //= 256
            binary += b
        return binary