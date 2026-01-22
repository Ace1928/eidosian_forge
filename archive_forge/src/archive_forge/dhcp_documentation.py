import random
import struct
import netaddr
from os_ken.lib import addrconv
from os_ken.lib import stringify
from . import packet_base
DHCP (RFC 2132) options encoder/decoder class.

    This is used with os_ken.lib.packet.dhcp.dhcp.options.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== ====================
    Attribute      Description
    ============== ====================
    tag            Option type.                   (except for the 'magic cookie', 'pad option'                    and 'end option'.)
    value          Option's value.                   (set the value that has been converted to hexadecimal.)
    length         Option's value length.                   (calculated automatically from the length of value.)
    ============== ====================
    