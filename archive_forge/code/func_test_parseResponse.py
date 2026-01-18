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
def test_parseResponse(self):
    """
        There should be a class method of Command which accepts a
        mapping of argument names to serialized forms and returns a
        similar mapping whose values have been parsed via the
        Command's response schema.
        """
    protocol = object()
    result = b'whatever'
    strings = {b'weird': result}
    self.assertEqual(ProtocolIncludingCommand.parseResponse(strings, protocol), {'weird': (result, protocol)})