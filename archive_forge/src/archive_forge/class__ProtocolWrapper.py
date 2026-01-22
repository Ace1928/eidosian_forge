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
class _ProtocolWrapper(protocol.ProcessProtocol):
    """
    This class wraps a L{Protocol} instance in a L{ProcessProtocol} instance.
    """

    def __init__(self, proto):
        self.proto = proto

    def connectionMade(self):
        self.proto.connectionMade()

    def outReceived(self, data):
        self.proto.dataReceived(data)

    def processEnded(self, reason):
        self.proto.connectionLost(reason)