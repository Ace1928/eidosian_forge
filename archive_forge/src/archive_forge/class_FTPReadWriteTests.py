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
class FTPReadWriteTests(TestCase, IReadWriteTestsMixin):
    """
    Tests for C{ftp._FileReader} and C{ftp._FileWriter}, the objects returned
    by the shell in C{openForReading}/C{openForWriting}.
    """

    def setUp(self):
        """
        Create a temporary file used later.
        """
        self.root = filepath.FilePath(self.mktemp())
        self.root.createDirectory()
        self.shell = ftp.FTPShell(self.root)
        self.filename = 'file.txt'

    def getFileReader(self, content):
        """
        Return a C{ftp._FileReader} instance with a file opened for reading.
        """
        self.root.child(self.filename).setContent(content)
        return self.shell.openForReading((self.filename,))

    def getFileWriter(self):
        """
        Return a C{ftp._FileWriter} instance with a file opened for writing.
        """
        return self.shell.openForWriting((self.filename,))

    def getFileContent(self):
        """
        Return the content of the temporary file.
        """
        return self.root.child(self.filename).getContent()