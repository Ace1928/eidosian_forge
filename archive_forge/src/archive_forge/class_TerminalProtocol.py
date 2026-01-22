from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
@implementer(ITerminalProtocol)
class TerminalProtocol:

    def makeConnection(self, terminal):
        self.terminal = terminal
        self.connectionMade()

    def connectionMade(self):
        """
        Called after a connection has been established.
        """

    def keystrokeReceived(self, keyID, modifier):
        pass

    def terminalSize(self, width, height):
        pass

    def unhandledControlSequence(self, seq):
        pass

    def connectionLost(self, reason):
        pass