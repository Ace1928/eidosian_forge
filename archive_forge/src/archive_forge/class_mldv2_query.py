import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
class mldv2_query(mld):
    """
    ICMPv6 sub encoder/decoder class for MLD v2 Lister Query messages.
    (RFC 3810)

    http://www.ietf.org/rfc/rfc3810.txt

    This is used with os_ken.lib.packet.icmpv6.icmpv6.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== =========================================
    Attribute      Description
    ============== =========================================
    maxresp        max response time in millisecond. it is
                   meaningful only in Query Message.
    address        a group address value.
    s_flg          when set to 1, routers suppress the timer
                   process.
    qrv            robustness variable for a querier.
    qqic           an interval time for a querier in unit of
                   seconds.
    num            a number of the multicast servers.
    srcs           a list of IPv6 addresses of the multicast
                   servers.
    ============== =========================================
    """
    _PACK_STR = '!H2x16sBBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': ['address'], 'asciilist': ['srcs']}

    def __init__(self, maxresp=0, address='::', s_flg=0, qrv=2, qqic=0, num=0, srcs=None):
        super(mldv2_query, self).__init__(maxresp, address)
        self.s_flg = s_flg
        self.qrv = qrv
        self.qqic = qqic
        self.num = num
        srcs = srcs or []
        assert isinstance(srcs, list)
        for src in srcs:
            assert isinstance(src, str)
        self.srcs = srcs

    @classmethod
    def parser(cls, buf):
        maxresp, address, s_qrv, qqic, num = struct.unpack_from(cls._PACK_STR, buf)
        s_flg = s_qrv >> 3 & 1
        qrv = s_qrv & 7
        offset = cls._MIN_LEN
        srcs = []
        while 0 < len(buf[offset:]) and num > len(srcs):
            assert 16 <= len(buf[offset:])
            src, = struct.unpack_from('16s', buf, offset)
            srcs.append(addrconv.ipv6.bin_to_text(src))
            offset += 16
        assert num == len(srcs)
        return cls(maxresp, addrconv.ipv6.bin_to_text(address), s_flg, qrv, qqic, num, srcs)

    def serialize(self):
        s_qrv = self.s_flg << 3 | self.qrv
        buf = bytearray(struct.pack(self._PACK_STR, self.maxresp, addrconv.ipv6.text_to_bin(self.address), s_qrv, self.qqic, self.num))
        for src in self.srcs:
            buf.extend(struct.pack('16s', addrconv.ipv6.text_to_bin(src)))
        if 0 == self.num:
            self.num = len(self.srcs)
            struct.pack_into('!H', buf, 22, self.num)
        return bytes(buf)

    def __len__(self):
        return self._MIN_LEN + len(self.srcs) * 16