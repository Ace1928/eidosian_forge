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
class SingleUseFactory(protocol.ClientFactory):

    def __init__(self, proto):
        self.proto = proto
        self.proto.factory = self

    def buildProtocol(self, addr):
        p, self.proto = (self.proto, None)
        return p
    reasonFailed = None

    def clientConnectionFailed(self, connector, reason):
        self.reasonFailed = reason
        return