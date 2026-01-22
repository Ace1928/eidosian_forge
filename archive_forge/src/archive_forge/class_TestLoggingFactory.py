import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
class TestLoggingFactory(policies.TrafficLoggingFactory):
    openFile = None

    def open(self, name):
        assert self.openFile is None, 'open() called too many times'
        self.openFile = StringIO()
        return self.openFile