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
def test_toBox(self):
    """
        L{ListOf.toBox} extracts the list of objects from the C{objects}
        dictionary passed to it, using the C{name} key also passed to it,
        serializes each of the elements in that list using the L{Argument}
        instance previously passed to its initializer, combines the serialized
        results, and inserts the result into the C{strings} dictionary using
        the same C{name} key.
        """
    stringList = amp.ListOf(self.elementType)
    strings = amp.AmpBox()
    for key in self.objects:
        stringList.toBox(key.encode('ascii'), strings, self.objects.copy(), None)
    self.assertEqual(strings, self.strings)