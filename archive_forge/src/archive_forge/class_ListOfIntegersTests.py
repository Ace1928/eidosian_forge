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
class ListOfIntegersTests(TestCase, ListOfTestsMixin):
    """
    Tests for L{ListOf} combined with L{amp.Integer}.
    """
    elementType = amp.Integer()
    huge = 9999999999999999999999999999999999999999999999999999999999 * 9999999999999999999999999999999999999999999999999999999999
    strings = {b'empty': b'', b'single': b'\x00\x0210', b'multiple': b'\x00\x011\x00\x0220\x00\x03500', b'huge': b'\x00t%d' % (huge,), b'negative': b'\x00\x02-1'}
    objects = {'empty': [], 'single': [10], 'multiple': [1, 20, 500], 'huge': [huge], 'negative': [-1]}