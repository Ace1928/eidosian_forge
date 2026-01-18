import numbers
import struct
import netaddr
from os_ken.lib import addrconv
from os_ken.lib import type_desc
def text_to_bin(ip):
    """
    Converts human readable IPv4 or IPv6 string to binary representation.
    :param str ip: IPv4 or IPv6 address string
    :return: binary representation of IPv4 or IPv6 address
    """
    if ':' not in ip:
        return ipv4_to_bin(ip)
    else:
        return ipv6_to_bin(ip)