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
Override method for the Length/Type field (self.ethertype).
        The Length/Type field means Length or Type interpretation,
        same as ethernet IEEE802.3.
        If the value of Length/Type field is less than or equal to
        1500 decimal(05DC hexadecimal), it means Length interpretation
        and be passed to the LLC sublayer.