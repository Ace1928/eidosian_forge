import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_sendUnhandledError(self):
    """
        L{CommandDispatcher} should relay its unhandled errors in responding to
        boxes to its boxSender.
        """
    err = RuntimeError('something went wrong, oh no')
    self.sender.expectError()
    self.dispatcher.unhandledError(Failure(err))
    self.assertEqual(len(self.sender.unhandledErrors), 1)
    self.assertEqual(self.sender.unhandledErrors[0].value, err)