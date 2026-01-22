import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import stringify
class _ICMPv4Payload(stringify.StringifyMixin, metaclass=abc.ABCMeta):
    """
    Base class for the payload of ICMPv4 packet.
    """