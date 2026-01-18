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
def test_statFile(self):
    """
        Check the output of the stat method on a file.
        """
    fileContent = b'wobble\n'
    self.createFile('file.txt', fileContent)
    d = self.shell.stat(('file.txt',), ('size', 'directory'))

    def cb(res):
        self.assertEqual(res[0], len(fileContent))
        self.assertFalse(res[1])
    d.addCallback(cb)
    return d