import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_setTimeout(self):
    """
        L{ftp.DTPFactory.setTimeout} uses the reactor passed to its initializer
        to set up a timed event to time out the DTP setup after the specified
        number of seconds.
        """
    finished = []
    d = self.assertFailure(self.factory.deferred, ftp.PortConnectionError)
    d.addCallback(finished.append)
    self.factory.setTimeout(6)
    self.reactor.advance(5)
    self.assertFalse(finished)
    self.reactor.advance(1)
    self.assertTrue(finished)
    self.assertFalse(self.reactor.calls)