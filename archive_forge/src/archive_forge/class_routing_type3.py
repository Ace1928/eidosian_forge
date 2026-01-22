import abc
import struct
from . import packet_base
from . import icmpv6
from . import tcp
from . import udp
from . import sctp
from . import gre
from . import in_proto as inet
from os_ken.lib import addrconv
from os_ken.lib import stringify
class routing_type3(header):
    """
    An IPv6 Routing Header for Source Routes with the RPL (RFC 6554)
    encoder/decoder class.

    This is used with os_ken.lib.packet.ipv6.ipv6.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =======================================
    Attribute      Description
    ============== =======================================
    nxt            Next Header
    size           The length of the Routing header,
                   not include the first 8 octet.
                   (0 means automatically-calculate when encoding)
    type           Identifies the particular Routing header variant.
    seg            Number of route segments remaining.
    cmpi           Number of prefix octets from segments 1 through n-1.
    cmpe           Number of prefix octets from segment n.
    pad            Number of octets that are used for padding
                   after Address[n] at the end of the SRH.
    adrs           Vector of addresses, numbered 1 to n.
    ============== =======================================
    """
    _PACK_STR = '!BBBBBB2x'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'asciilist': ['adrs']}

    def __init__(self, nxt=inet.IPPROTO_TCP, size=0, type_=3, seg=0, cmpi=0, cmpe=0, adrs=None):
        super(routing_type3, self).__init__(nxt)
        self.size = size
        self.type_ = type_
        self.seg = seg
        self.cmpi = cmpi
        self.cmpe = cmpe
        adrs = adrs or []
        assert isinstance(adrs, list)
        self.adrs = adrs
        self._pad = (8 - ((len(self.adrs) - 1) * (16 - self.cmpi) + (16 - self.cmpe) % 8)) % 8

    @classmethod
    def _get_size(cls, size):
        return (int(size) + 1) * 8

    @classmethod
    def parser(cls, buf):
        nxt, size, type_, seg, cmp_, pad = struct.unpack_from(cls._PACK_STR, buf)
        data = cls._MIN_LEN
        header_len = cls._get_size(size)
        cmpi = int(cmp_ >> 4)
        cmpe = int(cmp_ & 15)
        pad = int(pad >> 4)
        adrs = []
        if size:
            adrs_len_i = 16 - cmpi
            adrs_len_e = 16 - cmpe
            form_i = '%ds' % adrs_len_i
            form_e = '%ds' % adrs_len_e
            while data < header_len - (adrs_len_e + pad):
                adr, = struct.unpack_from(form_i, buf[data:])
                adr = b'\x00' * cmpi + adr
                adrs.append(addrconv.ipv6.bin_to_text(adr))
                data += adrs_len_i
            adr, = struct.unpack_from(form_e, buf[data:])
            adr = b'\x00' * cmpe + adr
            adrs.append(addrconv.ipv6.bin_to_text(adr))
        return cls(nxt, size, type_, seg, cmpi, cmpe, adrs)

    def serialize(self):
        if self.size == 0:
            self.size = ((len(self.adrs) - 1) * (16 - self.cmpi) + (16 - self.cmpe) + self._pad) // 8
        buf = struct.pack(self._PACK_STR, self.nxt, self.size, self.type_, self.seg, self.cmpi << 4 | self.cmpe, self._pad << 4)
        buf = bytearray(buf)
        if self.size:
            form_i = '%ds' % (16 - self.cmpi)
            form_e = '%ds' % (16 - self.cmpe)
            slice_i = slice(self.cmpi, 16)
            slice_e = slice(self.cmpe, 16)
            for adr in self.adrs[:-1]:
                buf.extend(struct.pack(form_i, addrconv.ipv6.text_to_bin(adr)[slice_i]))
            buf.extend(struct.pack(form_e, addrconv.ipv6.text_to_bin(self.adrs[-1])[slice_e]))
        return buf

    def __len__(self):
        return routing_type3._get_size(self.size)