from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
class _DummyController:
    """
    A do-nothing DNS controller.  This is useful when all messages received
    will be responses to previously issued queries.  Anything else received
    will be ignored.
    """

    def messageReceived(self, *args):
        pass