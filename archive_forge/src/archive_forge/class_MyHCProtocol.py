import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
@implementer(IHalfCloseableProtocol)
class MyHCProtocol(AccumulatingProtocol):
    readHalfClosed = False
    writeHalfClosed = False

    def readConnectionLost(self):
        self.readHalfClosed = True
        if self.writeHalfClosed:
            self.connectionLost(None)

    def writeConnectionLost(self):
        self.writeHalfClosed = True
        if self.readHalfClosed:
            self.connectionLost(None)