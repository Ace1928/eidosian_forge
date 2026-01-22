import sys
import twisted.internet
from twisted.test.test_twisted import SetAsideModule
class AlternateReactor(NoReactor):
    """
    A context manager which temporarily installs a different object as the
    global reactor.
    """

    def __init__(self, reactor):
        """
        @param reactor: Any object to install as the global reactor.
        """
        NoReactor.__init__(self)
        self.alternate = reactor

    def __enter__(self):
        NoReactor.__enter__(self)
        twisted.internet.reactor = self.alternate
        sys.modules['twisted.internet.reactor'] = self.alternate