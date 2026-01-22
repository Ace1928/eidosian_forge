import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
class GoodClient(Server):

    def connectionRefused(self):
        if self.startedDeferred is not None:
            d, self.startedDeferred = (self.startedDeferred, None)
            d.errback(error.ConnectionRefusedError('yup'))
        self.refused = 1