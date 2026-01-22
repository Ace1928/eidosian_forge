from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
def _roots(self):
    """
        Return a list of two-tuples representing the addresses of the root
        servers, as defined by C{self.hints}.
        """
    return [(ip, dns.PORT) for ip in self.hints]