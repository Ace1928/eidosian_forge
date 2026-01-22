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
def _register_header_type(cls):
    ipv6._IPV6_EXT_HEADER_TYPE[type_] = cls
    return cls