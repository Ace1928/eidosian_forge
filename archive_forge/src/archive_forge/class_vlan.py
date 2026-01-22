import abc
import struct
from . import packet_base
from . import arp
from . import ipv4
from . import ipv6
from . import lldp
from . import slow
from . import llc
from . import pbb
from . import cfm
from . import ether_types as ether
class vlan(_vlan):
    """VLAN (IEEE 802.1Q) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== ====================
    Attribute      Description
    ============== ====================
    pcp            Priority Code Point
    cfi            Canonical Format Indicator
    vid            VLAN Identifier
    ethertype      EtherType
    ============== ====================
    """

    def __init__(self, pcp=0, cfi=0, vid=0, ethertype=ether.ETH_TYPE_IP):
        super(vlan, self).__init__(pcp, cfi, vid, ethertype)

    @classmethod
    def get_packet_type(cls, type_):
        """Override method for the Length/Type field (self.ethertype).
        The Length/Type field means Length or Type interpretation,
        same as ethernet IEEE802.3.
        If the value of Length/Type field is less than or equal to
        1500 decimal(05DC hexadecimal), it means Length interpretation
        and be passed to the LLC sublayer."""
        if type_ <= ether.ETH_TYPE_IEEE802_3:
            type_ = ether.ETH_TYPE_IEEE802_3
        return cls._TYPES.get(type_)