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
class ListOfDecimalNanTests(TestCase, ListOfTestsMixin):
    """
    Tests for L{ListOf} combined with L{amp.Decimal} for not-a-number values.
    """
    elementType = amp.Decimal()
    strings = {b'nan': b'\x00\x03NaN\x00\x04-NaN\x00\x04sNaN\x00\x05-sNaN'}
    objects = {'nan': [decimal.Decimal('NaN'), decimal.Decimal('-NaN'), decimal.Decimal('sNaN'), decimal.Decimal('-sNaN')]}

    def test_fromBox(self):
        """
        L{ListOf.fromBox} reverses the operation performed by L{ListOf.toBox}.
        """

        def is_qnan(decimal):
            return 'NaN' in str(decimal) and 'sNaN' not in str(decimal)

        def is_snan(decimal):
            return 'sNaN' in str(decimal)

        def is_signed(decimal):
            return '-' in str(decimal)
        stringList = amp.ListOf(self.elementType)
        objects = {}
        for key in self.strings:
            stringList.fromBox(key, self.strings.copy(), objects, None)
        n = objects['nan']
        self.assertTrue(is_qnan(n[0]) and (not is_signed(n[0])))
        self.assertTrue(is_qnan(n[1]) and is_signed(n[1]))
        self.assertTrue(is_snan(n[2]) and (not is_signed(n[2])))
        self.assertTrue(is_snan(n[3]) and is_signed(n[3]))