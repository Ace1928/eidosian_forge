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
def test_responderOverriding(self):
    """
        L{CommandLocator} subclasses can override a responder inherited from
        a base class by using the L{Command.responder} decorator to register
        a new responder method.
        """
    return self._checkSimpleGreeting(OverridingLocator, 9)