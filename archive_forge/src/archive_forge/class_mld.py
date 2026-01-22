import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
@icmpv6.register_icmpv6_type(MLD_LISTENER_QUERY, MLD_LISTENER_REPOR, MLD_LISTENER_DONE)
class mld(_ICMPv6Payload):
    """ICMPv6 sub encoder/decoder class for MLD Lister Query,
    MLD Listener Report, and MLD Listener Done messages. (RFC 2710)

    http://www.ietf.org/rfc/rfc2710.txt

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
    ============== =========================================
    """
    _PACK_STR = '!H2x16s'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': ['address']}

    def __init__(self, maxresp=0, address='::'):
        self.maxresp = maxresp
        self.address = address

    @classmethod
    def parser(cls, buf, offset):
        if cls._MIN_LEN < len(buf[offset:]):
            msg = mldv2_query.parser(buf[offset:])
        else:
            maxresp, address = struct.unpack_from(cls._PACK_STR, buf, offset)
            msg = cls(maxresp, addrconv.ipv6.bin_to_text(address))
        return msg

    def serialize(self):
        buf = struct.pack(mld._PACK_STR, self.maxresp, addrconv.ipv6.text_to_bin(self.address))
        return buf

    def __len__(self):
        return self._MIN_LEN