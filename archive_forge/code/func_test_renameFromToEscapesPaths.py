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
def test_renameFromToEscapesPaths(self):
    """
        L{ftp.FTPClient.rename} issues I{RNTO} and I{RNFR} commands with paths
        escaped according to U{http://cr.yp.to/ftp/filesystem.html}.
        """
    self._testLogin()
    fromFile = '/foo/ba\nr/baz'
    toFile = '/qu\nux'
    self.client.rename(fromFile, toFile)
    self.client.lineReceived(b'350 ')
    self.client.lineReceived(b'250 ')
    self.assertEqual(self.transport.value(), b'RNFR /foo/ba\x00r/baz\r\nRNTO /qu\x00ux\r\n')