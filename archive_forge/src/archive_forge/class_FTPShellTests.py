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
class FTPShellTests(TestCase, IFTPShellTestsMixin):
    """
    Tests for the C{ftp.FTPShell} object.
    """

    def setUp(self):
        """
        Create a root directory and instantiate a shell.
        """
        self.root = filepath.FilePath(self.mktemp())
        self.root.createDirectory()
        self.shell = ftp.FTPShell(self.root)

    def directoryExists(self, path):
        """
        Test if the directory exists at C{path}.
        """
        return self.root.child(path).isdir()

    def createDirectory(self, path):
        """
        Create a directory in C{path}.
        """
        return self.root.child(path).createDirectory()

    def fileExists(self, path):
        """
        Test if the file exists at C{path}.
        """
        return self.root.child(path).isfile()

    def createFile(self, path, fileContent=b''):
        """
        Create a file named C{path} with some content.
        """
        return self.root.child(path).setContent(fileContent)