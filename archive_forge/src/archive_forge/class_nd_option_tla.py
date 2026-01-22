import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
@nd_neighbor.register_nd_option_type
class nd_option_tla(nd_option_la):
    """ICMPv6 sub encoder/decoder class for Neighbor discovery
    Target Link-Layer Address Option. (RFC 4861)

    This is used with os_ken.lib.packet.icmpv6.nd_neighbor.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|p{35em}|

    ============== ====================
    Attribute      Description
    ============== ====================
    length         length of the option.                    (0 means automatically-calculate when encoding)
    hw_src         Link-Layer Address.                    NOTE: If the address is longer than 6 octets this contains                    the first 6 octets in the address.                    This implementation assumes the address has at least                    6 octets.
    data           A bytearray which contains the rest of Link-Layer Address                    and padding.  When encoding a packet, it's user's                    responsibility to provide necessary padding for 8-octets                    alignment required by the protocol.
    ============== ====================
    """

    @classmethod
    def option_type(cls):
        return ND_OPTION_TLA

    def __init__(self, length=0, hw_src='00:00:00:00:00:00', data=None):
        super(nd_option_tla, self).__init__(length, hw_src, data)