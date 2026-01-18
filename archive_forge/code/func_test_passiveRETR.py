import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_passiveRETR(self):
    """
        Test the RETR command in passive mode: get a file and verify its
        content.

        L{ftp.FTPClient.retrieveFile} should return a Deferred which fires
        with the protocol instance passed to it after the download has
        completed.

        (XXX - This API should be based on producers and consumers)
        """

    def cbRetr(res, proto):
        self.assertEqual(proto.buffer, b'x' * 1000)

    def cbConnect(host, port, factory):
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 12345)
        proto = factory.buildProtocol((host, port))
        proto.makeConnection(proto_helpers.StringTransport())
        self.client.lineReceived(b'150 File status okay; about to open data connection.')
        proto.dataReceived(b'x' * 1000)
        proto.connectionLost(failure.Failure(error.ConnectionDone('')))
    self.client.connectFactory = cbConnect
    self._testLogin()
    proto = _BufferingProtocol()
    d = self.client.retrieveFile('spam', proto)
    d.addCallback(cbRetr, proto)
    self.assertEqual(self.transport.value(), b'PASV\r\n')
    self.transport.clear()
    self.client.lineReceived(passivemode_msg(self.client))
    self.assertEqual(self.transport.value(), b'RETR spam\r\n')
    self.transport.clear()
    self.client.lineReceived(b'226 Transfer Complete.')
    return d