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
def _checkForCommand(self):
    prompt = b'cftp> '
    if self._expectingCommand and self._lineBuffer == prompt:
        buf = b'\n'.join(self._linesReceived)
        if buf.startswith(prompt):
            buf = buf[len(prompt):]
        self.clearBuffer()
        d, self._expectingCommand = (self._expectingCommand, None)
        d.callback(buf)