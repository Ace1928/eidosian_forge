import errno
import os
import struct
import warnings
from typing import Dict
from zope.interface import implementer
from twisted.conch.interfaces import ISFTPFile, ISFTPServer
from twisted.conch.ssh.common import NS, getNS
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
def _sendRequest(self, msg, data):
    """
        Send a request and return a deferred which waits for the result.

        @type msg: L{int}
        @param msg: The request type (e.g., C{FXP_READ}).

        @type data: L{bytes}
        @param data: The body of the request.
        """
    if not self.connected:
        return defer.fail(error.ConnectionLost())
    data = struct.pack('!L', self.counter) + data
    d = defer.Deferred()
    self.openRequests[self.counter] = d
    self.counter += 1
    self.sendPacket(msg, data)
    return d