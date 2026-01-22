import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def _usable_range(self):
    if self.size >= 4:
        first_usable_address = self.first + 1
        if self._module.version == 4:
            last_usable_address = self.last - 1
        else:
            last_usable_address = self.last
        return (first_usable_address, last_usable_address)
    else:
        return (self.first, self.last)