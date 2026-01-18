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
def test_oldSingleDigitDayOfMonth(self):
    """
        A file with a high-resolution timestamp which falls on a day of the
        month which can be represented by one decimal digit is formatted with
        one padding 0 to preserve the columns which come after it.
        """
    then = self.now - 60 * 60 * 24 * 31 * 7 + 60 * 60 * 24 * 5
    stat = os.stat_result((0, 0, 0, 0, 0, 0, 0, 0, then, 0))
    self.assertEqual(self._lsInTimezone('America/New_York', stat), '!---------    0 0        0               0 May 01  1973 foo')
    self.assertEqual(self._lsInTimezone('Pacific/Auckland', stat), '!---------    0 0        0               0 May 02  1973 foo')