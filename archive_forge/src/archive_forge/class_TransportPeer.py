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
class TransportPeer(amp.Argument):

    def retrieve(self, d, name, proto):
        return b''

    def fromStringProto(self, notAString, proto):
        return proto.transport.getPeer()

    def toBox(self, name, strings, objects, proto):
        return