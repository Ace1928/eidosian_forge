import struct
import time
from zope.interface import implementer
from twisted.internet import interfaces, protocol
class Who(protocol.Protocol):
    """
    Return list of active users (RFC 866)
    """

    def connectionMade(self):
        self.transport.write(self.getUsers())
        self.transport.loseConnection()

    def getUsers(self):
        """
        Return active users. Override in subclasses.
        """
        return b'root\r\n'