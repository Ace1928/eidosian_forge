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
class StringListMailboxTests(TestCase):
    """
    Tests for L{StringListMailbox}, an in-memory only implementation of
    L{pop3.IMailbox}.
    """

    def test_listOneMessage(self):
        """
        L{StringListMailbox.listMessages} returns the length of the message at
        the offset into the mailbox passed to it.
        """
        mailbox = mail.maildir.StringListMailbox(['abc', 'ab', 'a'])
        self.assertEqual(mailbox.listMessages(0), 3)
        self.assertEqual(mailbox.listMessages(1), 2)
        self.assertEqual(mailbox.listMessages(2), 1)

    def test_listAllMessages(self):
        """
        L{StringListMailbox.listMessages} returns a list of the lengths of all
        messages if not passed an index.
        """
        mailbox = mail.maildir.StringListMailbox(['a', 'abc', 'ab'])
        self.assertEqual(mailbox.listMessages(), [1, 3, 2])

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_getMessage(self):
        """
        L{StringListMailbox.getMessage} returns a file-like object from which
        the contents of the message at the given offset into the mailbox can be
        read.
        """
        mailbox = mail.maildir.StringListMailbox(['foo', 'real contents'])
        self.assertEqual(mailbox.getMessage(1).read(), 'real contents')

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def test_getUidl(self):
        """
        L{StringListMailbox.getUidl} returns a unique identifier for the
        message at the given offset into the mailbox.
        """
        mailbox = mail.maildir.StringListMailbox(['foo', 'bar'])
        self.assertNotEqual(mailbox.getUidl(0), mailbox.getUidl(1))

    def test_deleteMessage(self):
        """
        L{StringListMailbox.deleteMessage} marks a message for deletion causing
        further requests for its length to return 0.
        """
        mailbox = mail.maildir.StringListMailbox(['foo'])
        mailbox.deleteMessage(0)
        self.assertEqual(mailbox.listMessages(0), 0)
        self.assertEqual(mailbox.listMessages(), [0])

    def test_undeleteMessages(self):
        """
        L{StringListMailbox.undeleteMessages} causes any messages marked for
        deletion to be returned to their original state.
        """
        mailbox = mail.maildir.StringListMailbox(['foo'])
        mailbox.deleteMessage(0)
        mailbox.undeleteMessages()
        self.assertEqual(mailbox.listMessages(0), 3)
        self.assertEqual(mailbox.listMessages(), [3])

    def test_sync(self):
        """
        L{StringListMailbox.sync} causes any messages as marked for deletion to
        be permanently deleted.
        """
        mailbox = mail.maildir.StringListMailbox(['foo'])
        mailbox.deleteMessage(0)
        mailbox.sync()
        mailbox.undeleteMessages()
        self.assertEqual(mailbox.listMessages(0), 0)
        self.assertEqual(mailbox.listMessages(), [0])