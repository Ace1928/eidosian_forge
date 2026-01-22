import dns.exception
from ._compat import long
class UnknownRcode(dns.exception.DNSException):
    """A DNS rcode is unknown."""