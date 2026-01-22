import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
@icmpv6.register_icmpv6_type(ND_ROUTER_SOLICIT)
class nd_router_solicit(_ICMPv6Payload):
    """ICMPv6 sub encoder/decoder class for Router Solicitation messages.
    (RFC 4861)

    This is used with os_ken.lib.packet.icmpv6.icmpv6.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|p{35em}|

    ============== ====================
    Attribute      Description
    ============== ====================
    res            This field is unused.  It MUST be initialized to zero.
    option         a derived object of os_ken.lib.packet.icmpv6.nd_option                    or a bytearray. None if no options.
    ============== ====================
    """
    _PACK_STR = '!I'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _ND_OPTION_TYPES = {}

    @staticmethod
    def register_nd_option_type(*args):

        def _register_nd_option_type(cls):
            nd_router_solicit._ND_OPTION_TYPES[cls.option_type()] = cls
            return cls
        return _register_nd_option_type(args[0])

    def __init__(self, res=0, option=None):
        self.res = res
        self.option = option

    @classmethod
    def parser(cls, buf, offset):
        res, = struct.unpack_from(cls._PACK_STR, buf, offset)
        offset += cls._MIN_LEN
        option = None
        if len(buf) > offset:
            type_, length = struct.unpack_from('!BB', buf, offset)
            if length == 0:
                raise struct.error('Invalid length: {len}'.format(len=length))
            cls_ = cls._ND_OPTION_TYPES.get(type_)
            if cls_ is not None:
                option = cls_.parser(buf, offset)
            else:
                option = buf[offset:]
        msg = cls(res, option)
        return msg

    def serialize(self):
        hdr = bytearray(struct.pack(nd_router_solicit._PACK_STR, self.res))
        if self.option is not None:
            if isinstance(self.option, nd_option):
                hdr.extend(self.option.serialize())
            else:
                hdr.extend(self.option)
        return bytes(hdr)

    def __len__(self):
        length = self._MIN_LEN
        if self.option is not None:
            length += len(self.option)
        return length