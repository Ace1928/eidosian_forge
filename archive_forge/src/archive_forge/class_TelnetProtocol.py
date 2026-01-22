import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
@implementer(ITelnetProtocol)
class TelnetProtocol(protocol.Protocol):
    _log = Logger()

    def unhandledCommand(self, command, argument):
        pass

    def unhandledSubnegotiation(self, command, data):
        pass

    def enableLocal(self, option):
        pass

    def enableRemote(self, option):
        pass

    def disableLocal(self, option):
        pass

    def disableRemote(self, option):
        pass