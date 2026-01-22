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
class VirtualPOP3Tests(TestCase):

    def setUp(self):
        self.tmpdir = self.mktemp()
        self.S = mail.mail.MailService()
        self.D = mail.maildir.MaildirDirdbmDomain(self.S, self.tmpdir)
        self.D.addUser(b'user', b'password')
        self.S.addDomain('test.domain', self.D)
        portal = cred.portal.Portal(self.D)
        map(portal.registerChecker, self.D.getCredentialsCheckers())
        self.S.portals[''] = self.S.portals['test.domain'] = portal
        self.P = mail.protocols.VirtualPOP3()
        self.P.service = self.S
        self.P.magic = '<unit test magic>'

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testAuthenticateAPOP(self):
        resp = md5(self.P.magic + 'password').hexdigest()
        return self.P.authenticateUserAPOP('user', resp).addCallback(self._cbAuthenticateAPOP)

    def _cbAuthenticateAPOP(self, result):
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], pop3.IMailbox)
        self.assertTrue(pop3.IMailbox.providedBy(result[1]))
        result[2]()

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testAuthenticateIncorrectUserAPOP(self):
        resp = md5(self.P.magic + 'password').hexdigest()
        return self.assertFailure(self.P.authenticateUserAPOP('resu', resp), cred.error.UnauthorizedLogin)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testAuthenticateIncorrectResponseAPOP(self):
        resp = md5('wrong digest').hexdigest()
        return self.assertFailure(self.P.authenticateUserAPOP('user', resp), cred.error.UnauthorizedLogin)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testAuthenticatePASS(self):
        return self.P.authenticateUserPASS('user', 'password').addCallback(self._cbAuthenticatePASS)

    def _cbAuthenticatePASS(self, result):
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], pop3.IMailbox)
        self.assertTrue(pop3.IMailbox.providedBy(result[1]))
        result[2]()

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testAuthenticateBadUserPASS(self):
        return self.assertFailure(self.P.authenticateUserPASS('resu', 'password'), cred.error.UnauthorizedLogin)

    @skipIf(sys.version_info >= (3,), 'not ported to Python 3')
    def testAuthenticateBadPasswordPASS(self):
        return self.assertFailure(self.P.authenticateUserPASS('user', 'wrong password'), cred.error.UnauthorizedLogin)