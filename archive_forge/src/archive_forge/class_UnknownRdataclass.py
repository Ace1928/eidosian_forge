import re
import dns.exception
class UnknownRdataclass(dns.exception.DNSException):
    """A DNS class is unknown."""