from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
def __address_in_network(self, ip, net):
    """Return boolean if IP is in network."""
    if net:
        ipaddr = int(''.join(['%02x' % int(x) for x in ip.split('.')]), 16)
        netstr, bits = net.split('/')
        netaddr = int(''.join(['%02x' % int(x) for x in netstr.split('.')]), 16)
        mask = 4294967295 << 32 - int(bits) & 4294967295
        return ipaddr & mask == netaddr & mask
    return True