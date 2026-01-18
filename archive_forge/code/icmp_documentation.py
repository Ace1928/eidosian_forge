import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import stringify
ICMP sub encoder/decoder class for Time Exceeded Message.

    This is used with os_ken.lib.packet.icmp.icmp for
    ICMP Time Exceeded Message.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    [RFC4884] introduced 8-bit data length attribute.

    .. tabularcolumns:: |l|L|

    ============== ====================
    Attribute      Description
    ============== ====================
    data_len       data length
    data           Internet Header + leading octets of original datagram
    ============== ====================
    