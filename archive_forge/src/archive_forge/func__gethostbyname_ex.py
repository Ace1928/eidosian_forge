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
def _gethostbyname_ex(name):
    aliases = []
    addresses = []
    tuples = _getaddrinfo(name, 0, socket.AF_INET, socket.SOCK_STREAM, socket.SOL_TCP, socket.AI_CANONNAME)
    canonical = tuples[0][3]
    for item in tuples:
        addresses.append(item[4][0])
    return (canonical, aliases, addresses)