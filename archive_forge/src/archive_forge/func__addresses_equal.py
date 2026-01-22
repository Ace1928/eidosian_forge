from __future__ import generators
import errno
import select
import socket
import struct
import sys
import time
import dns.exception
import dns.inet
import dns.name
import dns.message
import dns.rcode
import dns.rdataclass
import dns.rdatatype
from ._compat import long, string_types, PY3
def _addresses_equal(af, a1, a2):
    try:
        n1 = dns.inet.inet_pton(af, a1[0])
        n2 = dns.inet.inet_pton(af, a2[0])
    except dns.exception.SyntaxError:
        return False
    return n1 == n2 and a1[1:] == a2[1:]