import errno
import os
import sys
import warnings
from os import close, pathsep, pipe, read
from socket import AF_INET, AF_INET6, SOL_SOCKET, error, socket
from struct import pack
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.error import ProcessDone
from twisted.internet.protocol import ProcessProtocol
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_shortsend(self):
    """
        L{sendmsg} returns the number of bytes which it was able to send.
        """
    message = b'x' * 1024 * 1024 * 16
    self.input.setblocking(False)
    sent = sendmsg(self.input, message)
    self.assertTrue(sent < len(message))
    received = recvmsg(self.output, len(message))
    self.assertEqual(len(received[0]), sent)