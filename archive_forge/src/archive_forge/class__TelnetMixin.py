import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import checkers
from twisted.conch.test import test_telnet
class _TelnetMixin(_BaseMixin):

    def setUp(self):
        recvlineServer = self.serverProtocol()
        insultsServer = TestInsultsServerProtocol(lambda: recvlineServer)
        telnetServer = telnet.TelnetTransport(lambda: insultsServer)
        clientTransport = LoopbackRelay(telnetServer)
        recvlineClient = NotifyingExpectableBuffer()
        insultsClient = TestInsultsClientProtocol(lambda: recvlineClient)
        telnetClient = telnet.TelnetTransport(lambda: insultsClient)
        serverTransport = LoopbackRelay(telnetClient)
        telnetClient.makeConnection(clientTransport)
        telnetServer.makeConnection(serverTransport)
        serverTransport.clearBuffer()
        clientTransport.clearBuffer()
        self.recvlineClient = recvlineClient
        self.telnetClient = telnetClient
        self.clientTransport = clientTransport
        self.serverTransport = serverTransport
        return recvlineClient.onConnection

    def _testwrite(self, data):
        self.telnetClient.write(data)