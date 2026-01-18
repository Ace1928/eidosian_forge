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
def test_unhandledSerializationError(self):
    """
        Errors during serialization ought to be relayed to the sender's
        unhandledError method.
        """
    err = RuntimeError('something undefined went wrong')

    def thunk(result):

        class BrokenBox(amp.Box):

            def _sendTo(self, proto):
                raise err
        return BrokenBox()
    self.locator.commands['hello'] = thunk
    input = amp.Box(_command='hello', _ask='test-command-id', hello='world')
    self.sender.expectError()
    self.dispatcher.ampBoxReceived(input)
    self.assertEqual(len(self.sender.unhandledErrors), 1)
    self.assertEqual(self.sender.unhandledErrors[0].value, err)