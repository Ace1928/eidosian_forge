import struct
from math import trunc
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
class igmpv3_query(igmp):
    """
    Internet Group Management Protocol(IGMP, RFC 3376)
    Membership Query message encoder/decoder class.

    http://www.ietf.org/rfc/rfc3376.txt

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    =============== ====================================================
    Attribute       Description
    =============== ====================================================
    msgtype         a message type for v3.
    maxresp         max response time in unit of 1/10 second.
    csum            a check sum value. 0 means automatically-calculate
                    when encoding.
    address         a group address value.
    s_flg           when set to 1, routers suppress the timer process.
    qrv             robustness variable for a querier.
    qqic            an interval time for a querier in unit of seconds.
    num             a number of the multicast servers.
    srcs            a list of IPv4 addresses of the multicast servers.
    =============== ====================================================
    """
    _PACK_STR = '!BBH4sBBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    MIN_LEN = _MIN_LEN
    _TYPE = {'ascii': ['address'], 'asciilist': ['srcs']}

    def __init__(self, msgtype=IGMP_TYPE_QUERY, maxresp=100, csum=0, address='0.0.0.0', s_flg=0, qrv=2, qqic=0, num=0, srcs=None):
        super(igmpv3_query, self).__init__(msgtype, maxresp, csum, address)
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
        msgtype, maxresp, csum, address, s_qrv, qqic, num = struct.unpack_from(cls._PACK_STR, buf)
        s_flg = s_qrv >> 3 & 1
        qrv = s_qrv & 7
        offset = cls._MIN_LEN
        srcs = []
        while 0 < len(buf[offset:]) and num > len(srcs):
            assert 4 <= len(buf[offset:])
            src, = struct.unpack_from('4s', buf, offset)
            srcs.append(addrconv.ipv4.bin_to_text(src))
            offset += 4
        assert num == len(srcs)
        return (cls(msgtype, maxresp, csum, addrconv.ipv4.bin_to_text(address), s_flg, qrv, qqic, num, srcs), None, buf[offset:])

    def serialize(self, payload, prev):
        s_qrv = self.s_flg << 3 | self.qrv
        buf = bytearray(struct.pack(self._PACK_STR, self.msgtype, trunc(self.maxresp), self.csum, addrconv.ipv4.text_to_bin(self.address), s_qrv, trunc(self.qqic), self.num))
        for src in self.srcs:
            buf.extend(struct.pack('4s', addrconv.ipv4.text_to_bin(src)))
        if 0 == self.num:
            self.num = len(self.srcs)
            struct.pack_into('!H', buf, 10, self.num)
        if 0 == self.csum:
            self.csum = packet_utils.checksum(buf)
            struct.pack_into('!H', buf, 2, self.csum)
        return bytes(buf)

    def __len__(self):
        return self._MIN_LEN + len(self.srcs) * 4