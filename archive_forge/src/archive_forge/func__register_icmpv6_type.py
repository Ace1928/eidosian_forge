import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
def _register_icmpv6_type(cls):
    for type_ in args:
        icmpv6._ICMPV6_TYPES[type_] = cls
    return cls