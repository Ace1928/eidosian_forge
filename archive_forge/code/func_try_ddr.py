import contextlib
import random
import socket
import sys
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
import dns._ddr
import dns.edns
import dns.exception
import dns.flags
import dns.inet
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.nameserver
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.rdtypes.svcbbase
import dns.reversename
import dns.tsig
def try_ddr(self, lifetime: float=5.0) -> None:
    """Try to update the resolver's nameservers using Discovery of Designated
        Resolvers (DDR).  If successful, the resolver will subsequently use
        DNS-over-HTTPS or DNS-over-TLS for future queries.

        *lifetime*, a float, is the maximum time to spend attempting DDR.  The default
        is 5 seconds.

        If the SVCB query is successful and results in a non-empty list of nameservers,
        then the resolver's nameservers are set to the returned servers in priority
        order.

        The current implementation does not use any address hints from the SVCB record,
        nor does it resolve addresses for the SCVB target name, rather it assumes that
        the bootstrap nameserver will always be one of the addresses and uses it.
        A future revision to the code may offer fuller support.  The code verifies that
        the bootstrap nameserver is in the Subject Alternative Name field of the
        TLS certficate.
        """
    try:
        expiration = time.time() + lifetime
        answer = self.resolve(dns._ddr._local_resolver_name, 'SVCB', lifetime=lifetime)
        timeout = dns.query._remaining(expiration)
        nameservers = dns._ddr._get_nameservers_sync(answer, timeout)
        if len(nameservers) > 0:
            self.nameservers = nameservers
    except Exception:
        pass