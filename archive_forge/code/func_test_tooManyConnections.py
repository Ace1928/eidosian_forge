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
def test_tooManyConnections(self):
    """
        When the connection limit is reached, the server should send an
        appropriate response
        """
    self.factory.connectionLimit = 1
    cc = protocol.ClientCreator(reactor, _BufferingProtocol)
    d = cc.connectTCP('127.0.0.1', self.port.getHost().port)

    @d.addCallback
    def gotClient(proto):
        return proto.d

    @d.addCallback
    def onConnectionLost(proto):
        self.assertEqual(b'421 Too many users right now, try again in a few minutes.\r\n', proto.buffer)
    return d