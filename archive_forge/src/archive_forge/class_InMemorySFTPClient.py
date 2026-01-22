import getpass
import locale
import operator
import os
import struct
import sys
import time
from io import BytesIO, TextIOWrapper
from unittest import skipIf
from zope.interface import implementer
from twisted.conch import ls
from twisted.conch.interfaces import ISFTPFile
from twisted.conch.test.test_filetransfer import FileTransferTestAvatar, SFTPTestBase
from twisted.cred import portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.internet.utils import getProcessOutputAndValue, getProcessValue
from twisted.python import log
from twisted.python.fakepwd import UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
class InMemorySFTPClient:
    """
    A L{filetransfer.FileTransferClient} which does filesystem operations in
    memory, without touching the local disc or the network interface.

    @ivar _availableFiles: File like objects which are available to the SFTP
        client.
    @type _availableFiles: L{FilesystemRegister}
    """

    def __init__(self, availableFiles):
        self.transport = InMemorySSHChannel(self)
        self.options = {'requests': 1, 'buffersize': 10}
        self._availableFiles = availableFiles

    def openFile(self, filename, flags, attrs):
        """
        @see: L{filetransfer.FileTransferClient.openFile}.

        Retrieve and remove cached file based on flags.
        """
        return self._availableFiles.pop(filename, flags)