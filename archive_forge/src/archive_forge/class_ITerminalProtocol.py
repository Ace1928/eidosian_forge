from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
class ITerminalProtocol(Interface):

    def makeConnection(transport):
        """
        Called with an L{ITerminalTransport} when a connection is established.
        """

    def keystrokeReceived(keyID, modifier):
        """
        A keystroke was received.

        Each keystroke corresponds to one invocation of this method.
        keyID is a string identifier for that key.  Printable characters
        are represented by themselves.  Control keys, such as arrows and
        function keys, are represented with symbolic constants on
        L{ServerProtocol}.
        """

    def terminalSize(width, height):
        """
        Called to indicate the size of the terminal.

        A terminal of 80x24 should be assumed if this method is not
        called.  This method might not be called for real terminals.
        """

    def unhandledControlSequence(seq):
        """
        Called when an unsupported control sequence is received.

        @type seq: L{str}
        @param seq: The whole control sequence which could not be interpreted.
        """

    def connectionLost(reason):
        """
        Called when the connection has been lost.

        reason is a Failure describing why.
        """