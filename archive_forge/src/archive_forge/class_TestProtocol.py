from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
@implementer(telnet.ITelnetProtocol)
class TestProtocol:
    localEnableable = ()
    remoteEnableable = ()

    def __init__(self):
        self.data = b''
        self.subcmd = []
        self.calls = []
        self.enabledLocal = []
        self.enabledRemote = []
        self.disabledLocal = []
        self.disabledRemote = []

    def makeConnection(self, transport):
        d = transport.negotiationMap = {}
        d[b'\x12'] = self.neg_TEST_COMMAND
        d = transport.commandMap = transport.commandMap.copy()
        for cmd in ('EOR', 'NOP', 'DM', 'BRK', 'IP', 'AO', 'AYT', 'EC', 'EL', 'GA'):
            d[getattr(telnet, cmd)] = lambda arg, cmd=cmd: self.calls.append(cmd)

    def dataReceived(self, data):
        self.data += data

    def connectionLost(self, reason):
        pass

    def neg_TEST_COMMAND(self, payload):
        self.subcmd = payload

    def enableLocal(self, option):
        if option in self.localEnableable:
            self.enabledLocal.append(option)
            return True
        return False

    def disableLocal(self, option):
        self.disabledLocal.append(option)

    def enableRemote(self, option):
        if option in self.remoteEnableable:
            self.enabledRemote.append(option)
            return True
        return False

    def disableRemote(self, option):
        self.disabledRemote.append(option)

    def connectionMade(self):
        pass

    def unhandledCommand(self, command, argument):
        pass

    def unhandledSubnegotiation(self, command, data):
        pass