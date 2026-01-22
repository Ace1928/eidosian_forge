from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class TestTelnet(telnet.Telnet):
    """
    A trivial extension of the telnet protocol class useful to unit tests.
    """

    def __init__(self):
        telnet.Telnet.__init__(self)
        self.events = []

    def applicationDataReceived(self, data):
        """
        Record the given data in C{self.events}.
        """
        self.events.append(('bytes', data))

    def unhandledCommand(self, command, data):
        """
        Record the given command in C{self.events}.
        """
        self.events.append(('command', command, data))

    def unhandledSubnegotiation(self, command, data):
        """
        Record the given subnegotiation command in C{self.events}.
        """
        self.events.append(('negotiate', command, data))