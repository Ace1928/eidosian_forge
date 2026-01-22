import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@chunk_abort.register_cause_code
@chunk_error.register_cause_code
class cause_out_of_resource(cause):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Out of Resource (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.chunk_abort
    - os_ken.lib.packet.sctp.chunk_error

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    length         length of this cause containing this header.
                   (0 means automatically-calculate when encoding)
    ============== =====================================================
    """

    @classmethod
    def cause_code(cls):
        return CCODE_OUT_OF_RESOURCE

    @classmethod
    def parser(cls, buf):
        _, length = struct.unpack_from(cls._PACK_STR, buf)
        return cls(length)