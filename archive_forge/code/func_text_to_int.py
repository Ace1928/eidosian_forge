import numbers
import struct
import netaddr
from os_ken.lib import addrconv
from os_ken.lib import type_desc
def text_to_int(ip):
    """
    Converts human readable IPv4 or IPv6 string to int type representation.
    :param str ip: IPv4 or IPv6 address string
    :return: int type representation of IPv4 or IPv6 address
    """
    if ':' not in ip:
        return ipv4_to_int(ip)
    else:
        return ipv6_to_int(ip)