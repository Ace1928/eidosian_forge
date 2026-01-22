import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def _cbTestUnix(self, ignored, factory, s):
    TestEcho.d = defer.Deferred()
    factory.line = None
    factory.d = defer.Deferred()
    s.startService()
    factory.d.addCallback(self.assertEqual, b'lalala')
    factory.d.addCallback(lambda x: s.stopService())
    factory.d.addCallback(lambda x: TestEcho.d)
    return factory.d