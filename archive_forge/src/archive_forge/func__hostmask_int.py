import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
@property
def _hostmask_int(self):
    """Same as self.hostmask, but in integer format"""
    return (1 << self._module.width - self._prefixlen) - 1