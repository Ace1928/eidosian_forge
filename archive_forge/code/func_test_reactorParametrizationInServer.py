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
def test_reactorParametrizationInServer(self):
    """
        L{internet._AbstractServer} supports a C{reactor} keyword argument
        that can be used to parametrize the reactor used to listen for
        connections.
        """
    reactor = MemoryReactor()
    factory = object()
    t = internet.TCPServer(1234, factory, reactor=reactor)
    t.startService()
    self.assertEqual(reactor.tcpServers.pop()[:2], (1234, factory))