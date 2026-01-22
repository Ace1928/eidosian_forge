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
class NoAnswer(dns.exception.DNSException):
    """The DNS response does not contain an answer to the question."""
    fmt = 'The DNS response does not contain an answer ' + 'to the question: {query}'
    supp_kwargs = {'response'}

    def _fmt_kwargs(self, **kwargs):
        return super(NoAnswer, self)._fmt_kwargs(query=kwargs['response'].question)