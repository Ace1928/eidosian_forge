import random
import struct
from . import packet_base
from os_ken.lib import addrconv
from os_ken.lib import stringify
DHCP (RFC 3315) options encoder/decoder class.

    This is used with os_ken.lib.packet.dhcp6.dhcp6.options.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    The format of DHCP options is::

         0                   1                   2                   3
         0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |          option-code          |           option-len          |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |                          option-data                          |
        |                      (option-len octets)                      |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    ============== ====================
    Attribute      Description
    ============== ====================
    option-code    An unsigned integer identifying the specific option                   type carried in this option.
    option-len     An unsigned integer giving the length of the                   option-data field in this option in octets.
    option-data    The data for the option; the format of this data                   depends on the definition of the option.
    ============== ====================
    