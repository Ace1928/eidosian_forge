import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class tlv(stringify.StringifyMixin, metaclass=abc.ABCMeta):
    _TYPE_LEN = 1
    _LENGTH_LEN = 2
    _TYPE = {'ascii': ['egress_id_mac', 'last_egress_id_mac', 'next_egress_id_mac', 'mac_address']}

    def __init__(self, length):
        self.length = length

    @classmethod
    @abc.abstractmethod
    def parser(cls, buf):
        pass

    @abc.abstractmethod
    def serialize(self):
        pass

    def __len__(self):
        return self.length + self._TYPE_LEN + self._LENGTH_LEN