import email.message
import email.parser
import errno
import glob
import io
import os
import pickle
import shutil
import signal
import sys
import tempfile
import textwrap
import time
from hashlib import md5
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.portal
import twisted.mail.alias
import twisted.mail.mail
import twisted.mail.maildir
import twisted.mail.protocols
import twisted.mail.relay
import twisted.mail.relaymanager
from twisted import cred, mail
from twisted.internet import address, defer, interfaces, protocol, reactor, task
from twisted.internet.defer import Deferred
from twisted.internet.error import (
from twisted.internet.testing import (
from twisted.mail import pop3, smtp
from twisted.mail.relaymanager import _AttemptManager
from twisted.names import dns
from twisted.names.dns import Record_CNAME, Record_MX, RRHeader
from twisted.names.error import DNSNameError
from twisted.python import failure, log
from twisted.python.filepath import FilePath
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
from twisted.names import client, common, server
@skipIf(platformType != 'posix', 'twisted.mail only works on posix')
class MaildirMessageTests(TestCase):
    """
    Tests for the file creating by the L{mail.maildir.MaildirMessage}.
    """

    def setUp(self):
        """
        Create and open a temporary file.
        """
        self.name = self.mktemp()
        self.final = self.mktemp()
        self.address = b'user@example.com'
        self.f = open(self.name, 'wb')
        self.addCleanup(self.f.close)
        self.fp = mail.maildir.MaildirMessage(self.address, self.f, self.name, self.final)

    def _finalName(self):
        """
        Search for the final file path.

        @rtype: L{str}
        @return: Final file path.
        """
        return glob.glob(f'{self.final},S=[0-9]*')[0]

    def test_finalName(self):
        """
        Send the EOM to the message and check that the final file name contains
        the correct file size and the temporary file has been closed and removed.
        """
        final = self.successResultOf(self.fp.eomReceived())
        self.assertEqual(final, f'{self.final},S={os.path.getsize(final)}')
        self.assertTrue(self.f.closed)
        self.assertFalse(os.path.exists(self.name))

    def test_contents(self):
        """
        Send a message contents and the EOM to the message and check that the
        final file contains the correct header and the message contents.
        """
        contents = b'first line\nsecond line\nthird line\n'
        for line in contents.splitlines():
            self.fp.lineReceived(line)
        final = self.successResultOf(self.fp.eomReceived())
        with open(final, 'rb') as f:
            self.assertEqual(f.read(), b'Delivered-To: %s\n%s' % (self.address, contents))

    def test_interrupted(self):
        """
        Check that the interrupted message transfer removes the temporary file
        and a doesn't create a final file.
        """
        contents = b'first line\nsecond line\n'
        for line in contents.splitlines():
            self.fp.lineReceived(line)
        self.fp.connectionLost()
        self.assertFalse(os.path.exists(self.name))
        self.assertRaises(IndexError, self._finalName)