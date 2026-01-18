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
def test_passiveNLST(self):
    """
        Test the NLST command.

        Like L{test_passiveNLST} but in the configuration where the server
        establishes the data connection to the client, rather than the other
        way around.
        """

    def cbList(res, proto):
        fls = proto.buffer.splitlines()
        expected = [b'foo', b'bar', b'baz']
        expected.sort()
        fls.sort()
        self.assertEqual(fls, expected)

    def cbConnect(host, port, factory):
        self.assertEqual(host, '127.0.0.1')
        self.assertEqual(port, 12345)
        proto = factory.buildProtocol((host, port))
        proto.makeConnection(proto_helpers.StringTransport())
        self.client.lineReceived(b'150 File status okay; about to open data connection.')
        proto.dataReceived(b'foo\r\n')
        proto.dataReceived(b'bar\r\n')
        proto.dataReceived(b'baz\r\n')
        proto.connectionLost(failure.Failure(error.ConnectionDone('')))
    self.client.connectFactory = cbConnect
    self._testLogin()
    lstproto = _BufferingProtocol()
    d = self.client.nlst('foo/bar', lstproto).addCallback(cbList, lstproto)
    self.assertEqual(self.transport.value(), b'PASV\r\n')
    self.transport.clear()
    self.client.lineReceived(passivemode_msg(self.client))
    self.assertEqual(self.transport.value(), b'NLST foo/bar\r\n')
    self.client.lineReceived(b'226 Transfer Complete.')
    return d