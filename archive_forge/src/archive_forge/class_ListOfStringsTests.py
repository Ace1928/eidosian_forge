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
class ListOfStringsTests(TestCase, ListOfTestsMixin):
    """
    Tests for L{ListOf} combined with L{amp.String}.
    """
    elementType = amp.String()
    strings = {b'empty': b'', b'single': b'\x00\x03foo', b'multiple': b'\x00\x03bar\x00\x03baz\x00\x04quux'}
    objects = {'empty': [], 'single': [b'foo'], 'multiple': [b'bar', b'baz', b'quux']}