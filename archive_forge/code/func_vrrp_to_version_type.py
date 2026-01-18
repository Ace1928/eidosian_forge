import struct
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ether_types as ether
from os_ken.lib.packet import in_proto as inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vlan
from os_ken.lib import addrconv
def vrrp_to_version_type(version, type_):
    return version << VRRP_VERSION_SHIFT | type_