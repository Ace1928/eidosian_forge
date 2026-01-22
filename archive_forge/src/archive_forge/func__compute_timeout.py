import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def _compute_timeout(self, start, lifetime=None):
    lifetime = self.lifetime if lifetime is None else lifetime
    now = time.time()
    duration = now - start
    if duration < 0:
        if duration < -1:
            raise Timeout(timeout=duration)
        else:
            now = start
    if duration >= lifetime:
        raise Timeout(timeout=duration)
    return min(lifetime - duration, self.timeout)