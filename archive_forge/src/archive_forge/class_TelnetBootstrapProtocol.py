import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
class TelnetBootstrapProtocol(TelnetProtocol, ProtocolTransportMixin):
    protocol = None

    def __init__(self, protocolFactory, *args, **kw):
        self.protocolFactory = protocolFactory
        self.protocolArgs = args
        self.protocolKwArgs = kw

    def connectionMade(self):
        self.transport.negotiationMap[NAWS] = self.telnet_NAWS
        self.transport.negotiationMap[LINEMODE] = self.telnet_LINEMODE
        for opt in (LINEMODE, NAWS, SGA):
            self.transport.do(opt).addErrback(lambda f: self._log.failure('Error do {opt!r}', f, opt=opt))
        for opt in (ECHO,):
            self.transport.will(opt).addErrback(lambda f: self._log.failure('Error setting will {opt!r}', f, opt=opt))
        self.protocol = self.protocolFactory(*self.protocolArgs, **self.protocolKwArgs)
        try:
            factory = self.factory
        except AttributeError:
            pass
        else:
            self.protocol.factory = factory
        self.protocol.makeConnection(self)

    def connectionLost(self, reason):
        if self.protocol is not None:
            try:
                self.protocol.connectionLost(reason)
            finally:
                del self.protocol

    def dataReceived(self, data):
        self.protocol.dataReceived(data)

    def enableLocal(self, opt):
        if opt == ECHO:
            return True
        elif opt == SGA:
            return True
        else:
            return False

    def enableRemote(self, opt):
        if opt == LINEMODE:
            self.transport.requestNegotiation(LINEMODE, MODE + LINEMODE_TRAPSIG)
            return True
        elif opt == NAWS:
            return True
        elif opt == SGA:
            return True
        else:
            return False

    def telnet_NAWS(self, data):
        if len(data) == 4:
            width, height = struct.unpack('!HH', b''.join(data))
            self.protocol.terminalProtocol.terminalSize(width, height)
        else:
            self._log.error('Wrong number of NAWS bytes: {nbytes}', nbytes=len(data))
    linemodeSubcommands = {LINEMODE_SLC: 'SLC'}

    def telnet_LINEMODE(self, data):
        pass

    def linemode_SLC(self, data):
        chunks = zip(*[iter(data)] * 3)
        for slcFunction, slcValue, slcWhat in chunks:
            ('SLC', ord(slcFunction), ord(slcValue), ord(slcWhat))