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
def _fmt_kwargs(self, **kwargs):
    srv_msgs = []
    for err in kwargs['errors']:
        srv_msgs.append('Server {} {} port {} answered {}'.format(err[0], 'TCP' if err[1] else 'UDP', err[2], err[3]))
    return super(NoNameservers, self)._fmt_kwargs(query=kwargs['request'].question, errors='; '.join(srv_msgs))