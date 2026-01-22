from twisted.internet import defer, endpoints
from twisted.mail import protocols
from twisted.mail.tap import Options, makeService
from twisted.python.usage import UsageError
from twisted.trial.unittest import TestCase
class SpyEndpoint:
    """
    SpyEndpoint remembers what factory it is told to listen with.
    """
    listeningWith = None

    def listen(self, factory):
        self.listeningWith = factory
        return defer.succeed(None)