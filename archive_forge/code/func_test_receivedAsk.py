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
def test_receivedAsk(self):
    """
        L{CommandDispatcher.ampBoxReceived} should locate the appropriate
        command in its responder lookup, based on the '_ask' key.
        """
    received = []

    def thunk(box):
        received.append(box)
        return amp.Box({'hello': 'goodbye'})
    input = amp.Box(_command='hello', _ask='test-command-id', hello='world')
    self.locator.commands['hello'] = thunk
    self.dispatcher.ampBoxReceived(input)
    self.assertEqual(received, [input])