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
def _check_kwargs(self, qnames, responses=None):
    if not isinstance(qnames, (list, tuple, set)):
        raise AttributeError('qnames must be a list, tuple or set')
    if len(qnames) == 0:
        raise AttributeError('qnames must contain at least one element')
    if responses is None:
        responses = {}
    elif not isinstance(responses, dict):
        raise AttributeError('responses must be a dict(qname=response)')
    kwargs = dict(qnames=qnames, responses=responses)
    return kwargs