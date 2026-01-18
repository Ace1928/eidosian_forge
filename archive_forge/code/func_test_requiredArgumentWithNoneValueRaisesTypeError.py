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
def test_requiredArgumentWithNoneValueRaisesTypeError(self):
    """
        L{ListOf.toBox} raises C{TypeError} when passed a value of L{None}
        for the argument.
        """
    stringList = amp.ListOf(amp.Integer())
    self.assertRaises(TypeError, stringList.toBox, b'omitted', amp.AmpBox(), {'omitted': None}, None)