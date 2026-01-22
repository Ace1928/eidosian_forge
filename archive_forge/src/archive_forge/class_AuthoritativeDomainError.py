from twisted.internet.defer import TimeoutError
class AuthoritativeDomainError(ValueError):
    """
    Indicates a lookup failed for a name for which this server is authoritative
    because there were no records matching the given C{name, class, type}
    triple.
    """