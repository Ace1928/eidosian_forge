import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
class IOPump:
    """
    Utility to pump data between clients and servers for protocol testing.

    Perhaps this is a utility worthy of being in protocol.py?
    """

    def __init__(self, client, server, clientIO, serverIO):
        self.client = client
        self.server = server
        self.clientIO = clientIO
        self.serverIO = serverIO

    def flush(self):
        """
        Pump until there is no more input or output or until L{stop} is called.
        This does not run any timers, so don't use it with any code that calls
        reactor.callLater.
        """
        self._stop = False
        timeout = time.time() + 5
        while not self._stop and self.pump():
            if time.time() > timeout:
                return

    def stop(self):
        """
        Stop a running L{flush} operation, even if data remains to be
        transferred.
        """
        self._stop = True

    def pump(self):
        """
        Move data back and forth.

        Returns whether any data was moved.
        """
        self.clientIO.seek(0)
        self.serverIO.seek(0)
        cData = self.clientIO.read()
        sData = self.serverIO.read()
        self.clientIO.seek(0)
        self.serverIO.seek(0)
        self.clientIO.truncate()
        self.serverIO.truncate()
        self.client.transport._checkProducer()
        self.server.transport._checkProducer()
        for byte in iterbytes(cData):
            self.server.dataReceived(byte)
        for byte in iterbytes(sData):
            self.client.dataReceived(byte)
        if cData or sData:
            return 1
        else:
            return 0