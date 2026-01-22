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
def _checkSimpleGreeting(self, locatorClass, expected):
    """
        Check that a locator of type C{locatorClass} finds a responder
        for command named I{simple} and that the found responder answers
        with the C{expected} result to a C{SimpleGreeting<"ni hao", 5>}
        command.
        """
    locator = locatorClass()
    responderCallable = locator.locateResponder(b'simple')
    result = responderCallable(amp.Box(greeting=b'ni hao', cookie=b'5'))

    def done(values):
        self.assertEqual(values, amp.AmpBox(cookieplus=b'%d' % (expected,)))
    return result.addCallback(done)