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
class AddressAliasTests(TestCase):
    """
    Tests for L{twisted.mail.alias.AddressAlias}.
    """

    def setUp(self):
        """
        Setup an L{AddressAlias}.
        """
        self.address = mail.smtp.Address('foo@bar')
        domains = {self.address.domain: DummyDomain(self.address)}
        self.alias = mail.alias.AddressAlias(self.address, domains, self.address)

    def test_createMessageReceiver(self):
        """
        L{createMessageReceiever} calls C{exists()} on the domain object
        which key matches the C{alias} passed to L{AddressAlias}.
        """
        self.assertTrue(self.alias.createMessageReceiver())

    def test_str(self):
        """
        The string presentation of L{AddressAlias} includes the alias.
        """
        self.assertEqual(str(self.alias), '<Address foo@bar>')

    def test_resolve(self):
        """
        L{resolve} will look for additional aliases when an C{aliasmap}
        dictionary is passed, and returns L{None} if none were found.
        """
        self.assertEqual(self.alias.resolve({self.address: 'bar'}), None)

    def test_resolveWithoutAliasmap(self):
        """
        L{resolve} returns L{None} when the alias could not be found in the
        C{aliasmap} and no L{mail.smtp.User} with this alias exists either.
        """
        self.assertEqual(self.alias.resolve({}), None)