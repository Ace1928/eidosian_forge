import random
from io import StringIO
import struct
import dns.exception
import dns.rdatatype
import dns.rdataclass
import dns.rdata
import dns.set
from ._compat import string_types
def update_ttl(self, ttl):
    """Perform TTL minimization.

        Set the TTL of the rdataset to be the lesser of the set's current
        TTL or the specified TTL.  If the set contains no rdatas, set the TTL
        to the specified TTL.

        *ttl*, an ``int``.
        """
    if len(self) == 0:
        self.ttl = ttl
    elif ttl < self.ttl:
        self.ttl = ttl