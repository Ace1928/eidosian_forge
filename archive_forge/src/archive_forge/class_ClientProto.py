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
class ClientProto(protocol.ConnectedDatagramProtocol):
    started = stopped = False
    gotback = None

    def __init__(self):
        self.deferredStarted = defer.Deferred()
        self.deferredGotBack = defer.Deferred()

    def stopProtocol(self):
        self.stopped = True

    def startProtocol(self):
        self.started = True
        self.deferredStarted.callback(None)

    def datagramReceived(self, data):
        self.gotback = data
        self.deferredGotBack.callback(None)