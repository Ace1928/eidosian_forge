from twisted.internet.defer import TimeoutError
class DomainError(ValueError):
    """
    Indicates a lookup failed because there were no records matching the given
    C{name, class, type} triple.
    """