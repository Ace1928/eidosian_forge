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
@skipIf(sys.version_info >= (3,), 'not ported to Python 3')
def test_requestAvatarId(self):
    """
        L{DirdbmDatabase.requestAvatarId} raises L{UnauthorizedLogin} if
        supplied with invalid user credentials.
        When called with valid credentials, L{requestAvatarId} returns
        the username associated with the supplied credentials.
        """
    self.D.addUser('user', 'password')
    database = self.D.getCredentialsCheckers()[0]
    creds = cred.credentials.UsernamePassword('user', 'wrong password')
    self.assertRaises(cred.error.UnauthorizedLogin, database.requestAvatarId, creds)
    creds = cred.credentials.UsernamePassword('user', 'password')
    self.assertEqual(database.requestAvatarId(creds), 'user')