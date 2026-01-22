from io import StringIO
import sys
import dns.exception
import dns.name
import dns.ttl
from ._compat import long, text_type, binary_type
class UngetBufferFull(dns.exception.DNSException):
    """An attempt was made to unget a token when the unget buffer was full."""