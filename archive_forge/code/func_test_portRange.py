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
def test_portRange(self):
    """
        L{FTP.passivePortRange} should determine the ports which
        L{FTP.getDTPPort} attempts to bind. If no port from that iterator can
        be bound, L{error.CannotListenError} should be raised, otherwise the
        first successful result from L{FTP.listenFactory} should be returned.
        """

    def listenFactory(portNumber, factory):
        if portNumber in (22032, 22033, 22034):
            raise error.CannotListenError('localhost', portNumber, 'error')
        return portNumber
    self.serverProtocol.listenFactory = listenFactory
    port = self.serverProtocol.getDTPPort(protocol.Factory())
    self.assertEqual(port, 0)
    self.serverProtocol.passivePortRange = range(22032, 65536)
    port = self.serverProtocol.getDTPPort(protocol.Factory())
    self.assertEqual(port, 22035)
    self.serverProtocol.passivePortRange = range(22032, 22035)
    self.assertRaises(error.CannotListenError, self.serverProtocol.getDTPPort, protocol.Factory())