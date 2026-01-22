import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class chunk(stringify.StringifyMixin, metaclass=abc.ABCMeta):
    _PACK_STR = '!BBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    @abc.abstractmethod
    def chunk_type(cls):
        pass

    @abc.abstractmethod
    def __init__(self, type_, length):
        self._type = type_
        self.length = length

    @classmethod
    @abc.abstractmethod
    def parser(cls, buf):
        pass

    def __len__(self):
        return self.length