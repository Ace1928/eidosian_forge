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
class ListOfDecimalTests(TestCase, ListOfTestsMixin):
    """
    Tests for L{ListOf} combined with L{amp.Decimal}.
    """
    elementType = amp.Decimal()
    strings = {b'empty': b'', b'single': b'\x00\x031.1', b'extreme': b'\x00\x08Infinity\x00\t-Infinity', b'scientist': b'\x00\x083.141E+5\x00\n0.00003141\x00\x083.141E-7\x00\t-3.141E+5\x00\x0b-0.00003141\x00\t-3.141E-7', b'engineer': b'\x00\x04' + decimal.Decimal('0e6').to_eng_string().encode('ascii') + b'\x00\x06' + decimal.Decimal('1.5E-9').to_eng_string().encode('ascii')}
    objects = {'empty': [], 'single': [decimal.Decimal('1.1')], 'extreme': [decimal.Decimal('Infinity'), decimal.Decimal('-Infinity')], 'scientist': [decimal.Decimal('3.141E5'), decimal.Decimal('3.141e-5'), decimal.Decimal('3.141E-7'), decimal.Decimal('-3.141e5'), decimal.Decimal('-3.141E-5'), decimal.Decimal('-3.141e-7')], 'engineer': [decimal.Decimal('0e6'), decimal.Decimal('1.5E-9')]}