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
class header(stringify.StringifyMixin, metaclass=abc.ABCMeta):
    """extension header abstract class."""

    def __init__(self, nxt):
        self.nxt = nxt

    @classmethod
    @abc.abstractmethod
    def parser(cls, buf):
        pass

    @abc.abstractmethod
    def serialize(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass