from twisted.internet.defer import TimeoutError
class DNSQueryRefusedError(DomainError):
    """
    Indicates a query failed with a result of C{twisted.names.dns.EREFUSED}.
    """