import dns.exception
from ._compat import long
class BadTTL(dns.exception.SyntaxError):
    """DNS TTL value is not well-formed."""