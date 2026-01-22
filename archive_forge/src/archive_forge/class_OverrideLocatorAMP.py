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
class OverrideLocatorAMP(amp.AMP):

    def __init__(self):
        amp.AMP.__init__(self)
        self.customResponder = object()
        self.expectations = {b'custom': self.customResponder}
        self.greetings = []

    def lookupFunction(self, name):
        """
        Override the deprecated lookupFunction function.
        """
        if name in self.expectations:
            result = self.expectations[name]
            return result
        else:
            return super().lookupFunction(name)

    def greetingResponder(self, greeting, cookie):
        self.greetings.append((greeting, cookie))
        return dict(cookieplus=cookie + 3)
    greetingResponder = SimpleGreeting.responder(greetingResponder)