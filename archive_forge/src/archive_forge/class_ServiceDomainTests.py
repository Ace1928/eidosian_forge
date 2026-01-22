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
class ServiceDomainTests(TestCase):

    def setUp(self):
        self.S = mail.mail.MailService()
        self.D = mail.protocols.DomainDeliveryBase(self.S, None)
        self.D.service = self.S
        self.D.protocolName = 'TEST'
        self.D.host = 'hostname'
        self.tmpdir = self.mktemp()
        domain = mail.maildir.MaildirDirdbmDomain(self.S, self.tmpdir)
        domain.addUser(b'user', b'password')
        self.S.addDomain('test.domain', domain)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def testAddAliasableDomain(self):
        """
        Test that adding an IAliasableDomain to a mail service properly sets
        up alias group references and such.
        """
        aliases = object()
        domain = StubAliasableDomain()
        self.S.aliases = aliases
        self.S.addDomain('example.com', domain)
        self.assertIdentical(domain.aliasGroup, aliases)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testReceivedHeader(self):
        hdr = self.D.receivedHeader(('remotehost', '123.232.101.234'), smtp.Address('<someguy@someplace>'), ['user@host.name'])
        fp = io.BytesIO(hdr)
        emailParser = email.parser.Parser()
        m = emailParser.parse(fp)
        self.assertEqual(len(m.items()), 1)
        self.assertIn('Received', m)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testValidateTo(self):
        user = smtp.User('user@test.domain', 'helo', None, 'wherever@whatever')
        return defer.maybeDeferred(self.D.validateTo, user).addCallback(self._cbValidateTo)

    def _cbValidateTo(self, result):
        self.assertTrue(callable(result))

    def testValidateToBadUsername(self):
        user = smtp.User('resu@test.domain', 'helo', None, 'wherever@whatever')
        return self.assertFailure(defer.maybeDeferred(self.D.validateTo, user), smtp.SMTPBadRcpt)

    def testValidateToBadDomain(self):
        user = smtp.User('user@domain.test', 'helo', None, 'wherever@whatever')
        return self.assertFailure(defer.maybeDeferred(self.D.validateTo, user), smtp.SMTPBadRcpt)

    def testValidateFrom(self):
        helo = ('hostname', '127.0.0.1')
        origin = smtp.Address('<user@hostname>')
        self.assertTrue(self.D.validateFrom(helo, origin) is origin)
        helo = ('hostname', '1.2.3.4')
        origin = smtp.Address('<user@hostname>')
        self.assertTrue(self.D.validateFrom(helo, origin) is origin)
        helo = ('hostname', '1.2.3.4')
        origin = smtp.Address('<>')
        self.assertTrue(self.D.validateFrom(helo, origin) is origin)
        self.assertRaises(smtp.SMTPBadSender, self.D.validateFrom, None, origin)