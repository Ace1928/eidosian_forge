import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
class ProcessProtocolTests(unittest.TestCase):
    """
    Tests for behavior provided by the process protocol base class,
    L{protocol.ProcessProtocol}.
    """

    def test_interface(self):
        """
        L{ProcessProtocol} implements L{IProcessProtocol}.
        """
        verifyObject(interfaces.IProcessProtocol, protocol.ProcessProtocol())

    def test_outReceived(self):
        """
        Verify that when stdout is delivered to
        L{ProcessProtocol.childDataReceived}, it is forwarded to
        L{ProcessProtocol.outReceived}.
        """
        received = []

        class OutProtocol(StubProcessProtocol):

            def outReceived(self, data):
                received.append(data)
        bytesToSend = b'bytes'
        p = OutProtocol()
        p.childDataReceived(1, bytesToSend)
        self.assertEqual(received, [bytesToSend])

    def test_errReceived(self):
        """
        Similar to L{test_outReceived}, but for stderr.
        """
        received = []

        class ErrProtocol(StubProcessProtocol):

            def errReceived(self, data):
                received.append(data)
        bytesToSend = b'bytes'
        p = ErrProtocol()
        p.childDataReceived(2, bytesToSend)
        self.assertEqual(received, [bytesToSend])

    def test_inConnectionLost(self):
        """
        Verify that when stdin close notification is delivered to
        L{ProcessProtocol.childConnectionLost}, it is forwarded to
        L{ProcessProtocol.inConnectionLost}.
        """
        lost = []

        class InLostProtocol(StubProcessProtocol):

            def inConnectionLost(self):
                lost.append(None)
        p = InLostProtocol()
        p.childConnectionLost(0)
        self.assertEqual(lost, [None])

    def test_outConnectionLost(self):
        """
        Similar to L{test_inConnectionLost}, but for stdout.
        """
        lost = []

        class OutLostProtocol(StubProcessProtocol):

            def outConnectionLost(self):
                lost.append(None)
        p = OutLostProtocol()
        p.childConnectionLost(1)
        self.assertEqual(lost, [None])

    def test_errConnectionLost(self):
        """
        Similar to L{test_inConnectionLost}, but for stderr.
        """
        lost = []

        class ErrLostProtocol(StubProcessProtocol):

            def errConnectionLost(self):
                lost.append(None)
        p = ErrLostProtocol()
        p.childConnectionLost(2)
        self.assertEqual(lost, [None])