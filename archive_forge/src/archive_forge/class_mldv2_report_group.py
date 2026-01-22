import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
class mldv2_report_group(stringify.StringifyMixin):
    """
    ICMPv6 sub encoder/decoder class for MLD v2 Lister Report Group
    Record messages. (RFC 3810)

    This is used with os_ken.lib.packet.icmpv6.mldv2_report.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    =============== ====================================================
    Attribute       Description
    =============== ====================================================
    type\\_          a group record type for v3.
    aux_len         the length of the auxiliary data in 32-bit words.
    num             a number of the multicast servers.
    address         a group address value.
    srcs            a list of IPv6 addresses of the multicast servers.
    aux             the auxiliary data.
    =============== ====================================================
    """
    _PACK_STR = '!BBH16s'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': ['address'], 'asciilist': ['srcs']}

    def __init__(self, type_=0, aux_len=0, num=0, address='::', srcs=None, aux=None):
        self.type_ = type_
        self.aux_len = aux_len
        self.num = num
        self.address = address
        srcs = srcs or []
        assert isinstance(srcs, list)
        for src in srcs:
            assert isinstance(src, str)
        self.srcs = srcs
        self.aux = aux

    @classmethod
    def parser(cls, buf):
        type_, aux_len, num, address = struct.unpack_from(cls._PACK_STR, buf)
        offset = cls._MIN_LEN
        srcs = []
        while 0 < len(buf[offset:]) and num > len(srcs):
            assert 16 <= len(buf[offset:])
            src, = struct.unpack_from('16s', buf, offset)
            srcs.append(addrconv.ipv6.bin_to_text(src))
            offset += 16
        assert num == len(srcs)
        aux = None
        if aux_len:
            aux, = struct.unpack_from('%ds' % (aux_len * 4), buf, offset)
        msg = cls(type_, aux_len, num, addrconv.ipv6.bin_to_text(address), srcs, aux)
        return msg

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv6.text_to_bin(self.address)))
        for src in self.srcs:
            buf.extend(struct.pack('16s', addrconv.ipv6.text_to_bin(src)))
        if 0 == self.num:
            self.num = len(self.srcs)
            struct.pack_into('!H', buf, 2, self.num)
        if self.aux is not None:
            mod = len(self.aux) % 4
            if mod:
                self.aux += bytearray(4 - mod)
                self.aux = bytes(self.aux)
            buf.extend(self.aux)
            if 0 == self.aux_len:
                self.aux_len = len(self.aux) // 4
                struct.pack_into('!B', buf, 1, self.aux_len)
        return bytes(buf)

    def __len__(self):
        return self._MIN_LEN + len(self.srcs) * 16 + self.aux_len * 4