import os
import socket
import sys
from unittest import skipIf
from twisted.internet import address, defer, error, interfaces, protocol, reactor, utils
from twisted.python import lockfile
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.test.test_tcp import MyClientFactory, MyServerFactory
from twisted.trial import unittest
def test_socketLocking(self):
    """
        L{IReactorUNIX.listenUNIX} raises L{error.CannotListenError} if passed
        the name of a file on which a server is already listening.
        """
    filename = self.mktemp()
    serverFactory = MyServerFactory()
    unixPort = reactor.listenUNIX(filename, serverFactory, wantPID=True)
    self.assertRaises(error.CannotListenError, reactor.listenUNIX, filename, serverFactory, wantPID=True)

    def stoppedListening(ign):
        unixPort = reactor.listenUNIX(filename, serverFactory, wantPID=True)
        return unixPort.stopListening()
    return unixPort.stopListening().addCallback(stoppedListening)