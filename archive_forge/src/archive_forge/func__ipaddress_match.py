import re
import sys
def _ipaddress_match(ipname, host_ip):
    """Exact matching of IP addresses.

    RFC 6125 explicitly doesn't define an algorithm for this
    (section 1.7.2 - "Out of Scope").
    """
    ip = ipaddress.ip_address(_to_unicode(ipname).rstrip())
    return ip == host_ip