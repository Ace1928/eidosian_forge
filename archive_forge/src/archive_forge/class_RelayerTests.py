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
class RelayerTests(TestCase):

    def setUp(self):
        self.tmpdir = self.mktemp()
        os.mkdir(self.tmpdir)
        self.messageFiles = []
        for i in range(10):
            name = os.path.join(self.tmpdir, 'body-%d' % (i,))
            with open(name + '-H', 'wb') as f:
                pickle.dump(['from-%d' % (i,), 'to-%d' % (i,)], f)
            f = open(name + '-D', 'w')
            f.write(name)
            f.seek(0, 0)
            self.messageFiles.append(name)
        self.R = mail.relay.RelayerMixin()
        self.R.loadMessages(self.messageFiles)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def testMailFrom(self):
        for i in range(10):
            self.assertEqual(self.R.getMailFrom(), 'from-%d' % (i,))
            self.R.sentMail(250, None, None, None, None)
        self.assertEqual(self.R.getMailFrom(), None)

    def testMailTo(self):
        for i in range(10):
            self.assertEqual(self.R.getMailTo(), ['to-%d' % (i,)])
            self.R.sentMail(250, None, None, None, None)
        self.assertEqual(self.R.getMailTo(), None)

    def testMailData(self):
        for i in range(10):
            name = os.path.join(self.tmpdir, 'body-%d' % (i,))
            self.assertEqual(self.R.getMailData().read(), name)
            self.R.sentMail(250, None, None, None, None)
        self.assertEqual(self.R.getMailData(), None)