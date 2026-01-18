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
def test_openForWriting(self):
    """
        Check that openForWriting returns an object providing C{ftp.IWriteFile}.
        """
    d = self.shell.openForWriting(('foo',))

    def cb1(res):
        self.assertTrue(ftp.IWriteFile.providedBy(res))
        return res.receive().addCallback(cb2)

    def cb2(res):
        self.assertTrue(IConsumer.providedBy(res))
    d.addCallback(cb1)
    return d