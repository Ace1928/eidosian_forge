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
class FakeLocator:
    """
    This is a fake implementation of the interface implied by
    L{CommandLocator}.
    """

    def __init__(self):
        """
        Remember the given keyword arguments as a set of responders.
        """
        self.commands = {}

    def locateResponder(self, commandName):
        """
        Look up and return a function passed as a keyword argument of the given
        name to the constructor.
        """
        return self.commands[commandName]