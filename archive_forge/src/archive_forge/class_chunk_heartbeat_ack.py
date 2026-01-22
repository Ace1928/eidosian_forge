import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@sctp.register_chunk_type
class chunk_heartbeat_ack(chunk_heartbeat_base):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Heartbeat Acknowledgement
    (HEARTBEAT ACK) chunk (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.sctp

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    flags          set to '0'. this field will be ignored.
    length         length of this chunk containing this header.
                   (0 means automatically-calculate when encoding)
    info           os_ken.lib.packet.sctp.param_heartbeat.
    ============== =====================================================
    """
    _RECOGNIZED_PARAMS = {}

    @staticmethod
    def register_param_type(*args):

        def _register_param_type(cls):
            chunk_heartbeat_ack._RECOGNIZED_PARAMS[cls.param_type()] = cls
            return cls
        return _register_param_type(args[0])

    @classmethod
    def chunk_type(cls):
        return TYPE_HEARTBEAT_ACK

    @classmethod
    def parser(cls, buf):
        return super(chunk_heartbeat_ack, cls).parser_base(buf, cls._RECOGNIZED_PARAMS)