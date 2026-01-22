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
class ListOfOptionalTests(TestCase):
    """
    Tests to ensure L{ListOf} AMP arguments can be omitted from AMP commands
    via the 'optional' flag.
    """

    def test_requiredArgumentWithNoneValueRaisesTypeError(self):
        """
        L{ListOf.toBox} raises C{TypeError} when passed a value of L{None}
        for the argument.
        """
        stringList = amp.ListOf(amp.Integer())
        self.assertRaises(TypeError, stringList.toBox, b'omitted', amp.AmpBox(), {'omitted': None}, None)

    def test_optionalArgumentWithNoneValueOmitted(self):
        """
        L{ListOf.toBox} silently omits serializing any argument with a
        value of L{None} that is designated as optional for the protocol.
        """
        stringList = amp.ListOf(amp.Integer(), optional=True)
        strings = amp.AmpBox()
        stringList.toBox(b'omitted', strings, {b'omitted': None}, None)
        self.assertEqual(strings, {})

    def test_requiredArgumentWithKeyMissingRaisesKeyError(self):
        """
        L{ListOf.toBox} raises C{KeyError} if the argument's key is not
        present in the objects dictionary.
        """
        stringList = amp.ListOf(amp.Integer())
        self.assertRaises(KeyError, stringList.toBox, b'ommited', amp.AmpBox(), {'someOtherKey': 0}, None)

    def test_optionalArgumentWithKeyMissingOmitted(self):
        """
        L{ListOf.toBox} silently omits serializing any argument designated
        as optional whose key is not present in the objects dictionary.
        """
        stringList = amp.ListOf(amp.Integer(), optional=True)
        stringList.toBox(b'ommited', amp.AmpBox(), {b'someOtherKey': 0}, None)

    def test_omittedOptionalArgumentDeserializesAsNone(self):
        """
        L{ListOf.fromBox} correctly reverses the operation performed by
        L{ListOf.toBox} for optional arguments.
        """
        stringList = amp.ListOf(amp.Integer(), optional=True)
        objects = {}
        stringList.fromBox(b'omitted', {}, objects, None)
        self.assertEqual(objects, {'omitted': None})