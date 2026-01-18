import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
def test_threadsAreRunInScheduledOrder(self):
    """
        Callbacks should be invoked in the order they were scheduled.
        """
    order = []

    def check(_):
        self.assertEqual(order, [1, 2, 3])
    self.deferred.addCallback(check)
    self.schedule(order.append, 1)
    self.schedule(order.append, 2)
    self.schedule(order.append, 3)
    self.schedule(reactor.callFromThread, self.deferred.callback, None)
    return self.deferred