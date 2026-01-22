import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
class StatefulTelnetProtocol(basic.LineReceiver, TelnetProtocol):
    delimiter = b'\n'
    state = 'Discard'

    def connectionLost(self, reason):
        basic.LineReceiver.connectionLost(self, reason)
        TelnetProtocol.connectionLost(self, reason)

    def lineReceived(self, line):
        oldState = self.state
        newState = getattr(self, 'telnet_' + oldState)(line)
        if newState is not None:
            if self.state == oldState:
                self.state = newState
            else:
                self._log.warn('state changed and new state returned')

    def telnet_Discard(self, line):
        pass