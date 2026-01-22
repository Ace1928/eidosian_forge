import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
def _handleParseError(self):
    """
        Terminates the connection and sets the flag C{self.brokenPeer}.
        """
    self.transport.loseConnection()
    self.brokenPeer = 1