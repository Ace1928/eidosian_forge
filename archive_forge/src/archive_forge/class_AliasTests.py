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
class AliasTests(TestCase):
    lines = ['First line', 'Next line', '', 'After a blank line', 'Last line']

    def testHandle(self):
        result = {}
        lines = ['user:  another@host\n', 'nextuser:  |/bin/program\n', 'user:  me@again\n', 'moreusers: :/etc/include/filename\n', 'multiuser: first@host, second@host,last@anotherhost']
        for l in lines:
            mail.alias.handle(result, l, 'TestCase', None)
        self.assertEqual(result['user'], ['another@host', 'me@again'])
        self.assertEqual(result['nextuser'], ['|/bin/program'])
        self.assertEqual(result['moreusers'], [':/etc/include/filename'])
        self.assertEqual(result['multiuser'], ['first@host', 'second@host', 'last@anotherhost'])

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testFileLoader(self):
        domains = {'': object()}
        result = mail.alias.loadAliasFile(domains, fp=io.BytesIO(textwrap.dedent("                    # Here's a comment\n                       # woop another one\n                    testuser:                   address1,address2, address3,\n                        continuation@address, |/bin/process/this\n\n                    usertwo:thisaddress,thataddress, lastaddress\n                    lastuser:       :/includable, /filename, |/program, address\n                    ").encode()))
        self.assertEqual(len(result), 3)
        group = result['testuser']
        s = str(group)
        for a in ('address1', 'address2', 'address3', 'continuation@address', '/bin/process/this'):
            self.assertNotEqual(s.find(a), -1)
        self.assertEqual(len(group), 5)
        group = result['usertwo']
        s = str(group)
        for a in ('thisaddress', 'thataddress', 'lastaddress'):
            self.assertNotEqual(s.find(a), -1)
        self.assertEqual(len(group), 3)
        group = result['lastuser']
        s = str(group)
        self.assertEqual(s.find('/includable'), -1)
        for a in ('/filename', 'program', 'address'):
            self.assertNotEqual(s.find(a), -1, '%s not found' % a)
        self.assertEqual(len(group), 3)

    def testMultiWrapper(self):
        msgs = (LineBufferMessage(), LineBufferMessage(), LineBufferMessage())
        msg = mail.alias.MultiWrapper(msgs)
        for L in self.lines:
            msg.lineReceived(L)
        return msg.eomReceived().addCallback(self._cbMultiWrapper, msgs)

    def _cbMultiWrapper(self, ignored, msgs):
        for m in msgs:
            self.assertTrue(m.eom)
            self.assertFalse(m.lost)
            self.assertEqual(self.lines, m.lines)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testFileAlias(self):
        tmpfile = self.mktemp()
        a = mail.alias.FileAlias(tmpfile, None, None)
        m = a.createMessageReceiver()
        for l in self.lines:
            m.lineReceived(l)
        return m.eomReceived().addCallback(self._cbTestFileAlias, tmpfile)

    def _cbTestFileAlias(self, ignored, tmpfile):
        with open(tmpfile) as f:
            lines = f.readlines()
        self.assertEqual([L[:-1] for L in lines], self.lines)