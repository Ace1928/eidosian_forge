import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
def _cbTestPartialRetrieveWithConsumer(self, result, f, c):
    self.assertIdentical(result, f)
    self.assertEqual(c.data, [b'Line the first!  Woop', b'Line the last!  Bye'])