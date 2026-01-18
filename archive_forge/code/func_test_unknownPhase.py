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
def test_unknownPhase(self):
    """
        L{IReactorCore.addSystemEventTrigger} should reject phases other than
        C{'before'}, C{'during'}, or C{'after'}.
        """
    eventType = 'test'
    self.assertRaises(KeyError, self.addTrigger, 'xxx', eventType, lambda: None)