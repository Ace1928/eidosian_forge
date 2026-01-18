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
def test_putOverLongerFile(self):
    """
        Check that 'put' uploads files correctly when overwriting a longer
        file.
        """
    with self.testDir.child('shorterFile').open(mode='w') as f:
        f.write(b'a')
    with self.testDir.child('longerFile').open(mode='w') as f:
        f.write(b'bb')

    def _checkPut(result):
        self.assertFilesEqual(self.testDir.child('shorterFile'), self.testDir.child('longerFile'))
    d = self.runCommand(f'put {self.testDir.path}/shorterFile longerFile')
    d.addCallback(_checkPut)
    return d