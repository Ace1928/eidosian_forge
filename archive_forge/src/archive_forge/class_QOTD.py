import struct
import time
from zope.interface import implementer
from twisted.internet import interfaces, protocol
class QOTD(protocol.Protocol):
    """
    Return a quote of the day (RFC 865).
    """

    def connectionMade(self):
        self.transport.write(self.getQuote())
        self.transport.loseConnection()

    def getQuote(self):
        """
        Return a quote. May be overrriden in subclasses.
        """
        return b'An apple a day keeps the doctor away.\r\n'