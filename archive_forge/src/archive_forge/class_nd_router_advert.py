import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
@icmpv6.register_icmpv6_type(ND_ROUTER_ADVERT)
class nd_router_advert(_ICMPv6Payload):
    """ICMPv6 sub encoder/decoder class for Router Advertisement messages.
    (RFC 4861)

    This is used with os_ken.lib.packet.icmpv6.icmpv6.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|p{35em}|

    ============== ====================
    Attribute      Description
    ============== ====================
    ch_l           Cur Hop Limit.
    res            M,O Flags for Router Advertisement.
    rou_l          Router Lifetime.
    rea_t          Reachable Time.
    ret_t          Retrans Timer.
    options        List of a derived object of                    os_ken.lib.packet.icmpv6.nd_option or a bytearray.                    None if no options.
    ============== ====================
    """
    _PACK_STR = '!BBHII'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _ND_OPTION_TYPES = {}

    @staticmethod
    def register_nd_option_type(*args):

        def _register_nd_option_type(cls):
            nd_router_advert._ND_OPTION_TYPES[cls.option_type()] = cls
            return cls
        return _register_nd_option_type(args[0])

    def __init__(self, ch_l=0, res=0, rou_l=0, rea_t=0, ret_t=0, options=None):
        self.ch_l = ch_l
        self.res = res
        self.rou_l = rou_l
        self.rea_t = rea_t
        self.ret_t = ret_t
        options = options or []
        assert isinstance(options, list)
        self.options = options

    @classmethod
    def parser(cls, buf, offset):
        ch_l, res, rou_l, rea_t, ret_t = struct.unpack_from(cls._PACK_STR, buf, offset)
        offset += cls._MIN_LEN
        options = []
        while len(buf) > offset:
            type_, length = struct.unpack_from('!BB', buf, offset)
            if length == 0:
                raise struct.error('Invalid length: {len}'.format(len=length))
            cls_ = cls._ND_OPTION_TYPES.get(type_)
            if cls_ is not None:
                option = cls_.parser(buf, offset)
            else:
                option = buf[offset:offset + length * 8]
            options.append(option)
            offset += len(option)
        msg = cls(ch_l, res >> 6, rou_l, rea_t, ret_t, options)
        return msg

    def serialize(self):
        res = self.res << 6
        hdr = bytearray(struct.pack(nd_router_advert._PACK_STR, self.ch_l, res, self.rou_l, self.rea_t, self.ret_t))
        for option in self.options:
            if isinstance(option, nd_option):
                hdr.extend(option.serialize())
            else:
                hdr.extend(option)
        return bytes(hdr)

    def __len__(self):
        length = self._MIN_LEN
        for option in self.options:
            length += len(option)
        return length