from struct import calcsize, pack, unpack
from twisted.protocols.stateful import StatefulProtocol
from twisted.protocols.test import test_basic
from twisted.trial.unittest import TestCase
class MyInt32StringReceiver(StatefulProtocol):
    """
    A stateful Int32StringReceiver.
    """
    MAX_LENGTH = 99999
    structFormat = '!I'
    prefixLength = calcsize(structFormat)

    def getInitialState(self):
        return (self._getHeader, 4)

    def lengthLimitExceeded(self, length):
        self.transport.loseConnection()

    def _getHeader(self, msg):
        length, = unpack('!i', msg)
        if length > self.MAX_LENGTH:
            self.lengthLimitExceeded(length)
            return
        return (self._getString, length)

    def _getString(self, msg):
        self.stringReceived(msg)
        return (self._getHeader, 4)

    def stringReceived(self, msg):
        """
        Override this.
        """
        raise NotImplementedError

    def sendString(self, data):
        """
        Send an int32-prefixed string to the other end of the connection.
        """
        self.transport.write(pack(self.structFormat, len(data)) + data)