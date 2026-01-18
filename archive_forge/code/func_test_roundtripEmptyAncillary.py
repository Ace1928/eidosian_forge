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
def test_roundtripEmptyAncillary(self):
    """
        L{sendmsg} treats an empty ancillary data list the same way it treats
        receiving no argument for the ancillary parameter at all.
        """
    sendmsg(self.input, b'hello, world!', [], 0)
    result = recvmsg(self.output)
    self.assertEqual(result, (b'hello, world!', [], 0))