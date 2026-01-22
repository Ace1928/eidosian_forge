import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
class SSHSessionClient(protocol.Protocol):

    def dataReceived(self, data):
        if self.transport:
            self.transport.write(data)