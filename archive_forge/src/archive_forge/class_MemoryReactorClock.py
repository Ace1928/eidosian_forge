from __future__ import annotations
from io import BytesIO
from socket import AF_INET, AF_INET6
from typing import Callable, Iterator, Sequence, overload
from zope.interface import implementedBy, implementer
from zope.interface.verify import verifyClass
from typing_extensions import ParamSpec, Self
from twisted.internet import address, error, protocol, task
from twisted.internet.abstract import _dataMustBeBytes, isIPv6Address
from twisted.internet.address import IPv4Address, IPv6Address, UNIXAddress
from twisted.internet.defer import Deferred
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import (
from twisted.internet.task import Clock
from twisted.logger import ILogObserver, LogEvent, LogPublisher
from twisted.protocols import basic
from twisted.python import failure
from twisted.trial.unittest import TestCase
class MemoryReactorClock(MemoryReactor, Clock):

    def __init__(self):
        MemoryReactor.__init__(self)
        Clock.__init__(self)