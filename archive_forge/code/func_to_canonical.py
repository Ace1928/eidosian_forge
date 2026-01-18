import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def to_canonical(self):
    """
        Converts the address to IPv4 if it is an IPv4-mapped IPv6 address (`RFC 4291
        Section 2.5.5.2 <https://datatracker.ietf.org/doc/html/rfc4291.html#section-2.5.5.2>`_),
        otherwise returns the address as-is.

        >>> # IPv4-mapped IPv6
        >>> IPAddress('::ffff:10.0.0.1').to_canonical()
        IPAddress('10.0.0.1')
        >>>
        >>> # Everything else
        >>> IPAddress('::1').to_canonical()
        IPAddress('::1')
        >>> IPAddress('10.0.0.1').to_canonical()
        IPAddress('10.0.0.1')

        .. versionadded:: 0.10.0
        """
    if not self.is_ipv4_mapped():
        return self
    return self.ipv4()