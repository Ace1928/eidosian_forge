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
def test_removeNonExistentSystemEventTrigger(self):
    """
        Passing an object to L{IReactorCore.removeSystemEventTrigger} which was
        not returned by a previous call to
        L{IReactorCore.addSystemEventTrigger} or which has already been passed
        to C{removeSystemEventTrigger} should result in L{TypeError},
        L{KeyError}, or L{ValueError} being raised.
        """
    b = self.addTrigger('during', 'test', lambda: None)
    self.removeTrigger(b)
    self.assertRaises(TypeError, reactor.removeSystemEventTrigger, None)
    self.assertRaises(ValueError, reactor.removeSystemEventTrigger, b)
    self.assertRaises(KeyError, reactor.removeSystemEventTrigger, (b[0], ('xxx',) + b[1][1:]))