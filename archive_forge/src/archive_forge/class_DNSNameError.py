from twisted.internet.defer import TimeoutError
class DNSNameError(DomainError):
    """
    Indicates a query failed with a result of C{twisted.names.dns.ENAME}.
    """