import os
import sys
from enum import Enum, _simple_enum
def _arp_getnode():
    """Get the hardware address on Unix by running arp."""
    import os, socket
    if not hasattr(socket, 'gethostbyname'):
        return None
    try:
        ip_addr = socket.gethostbyname(socket.gethostname())
    except OSError:
        return None
    mac = _find_mac_near_keyword('arp', '-an', [os.fsencode(ip_addr)], lambda i: -1)
    if mac:
        return mac
    mac = _find_mac_near_keyword('arp', '-an', [os.fsencode(ip_addr)], lambda i: i + 1)
    if mac:
        return mac
    mac = _find_mac_near_keyword('arp', '-an', [os.fsencode('(%s)' % ip_addr)], lambda i: i + 2)
    if mac:
        return mac
    return None