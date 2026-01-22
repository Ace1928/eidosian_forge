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
@implementer(IConnector)
class _FakeConnector:
    """
    A fake L{IConnector} that allows us to inspect if it has been told to stop
    connecting.

    @ivar stoppedConnecting: has this connector's
        L{_FakeConnector.stopConnecting} method been invoked yet?

    @ivar _address: An L{IAddress} provider that represents our destination.
    """
    _disconnected = False
    stoppedConnecting = False

    def __init__(self, address):
        """
        @param address: An L{IAddress} provider that represents this
            connector's destination.
        """
        self._address = address

    def stopConnecting(self):
        """
        Implement L{IConnector.stopConnecting} and set
        L{_FakeConnector.stoppedConnecting} to C{True}
        """
        self.stoppedConnecting = True

    def disconnect(self):
        """
        Implement L{IConnector.disconnect} as a no-op.
        """
        self._disconnected = True

    def connect(self):
        """
        Implement L{IConnector.connect} as a no-op.
        """

    def getDestination(self):
        """
        Implement L{IConnector.getDestination} to return the C{address} passed
        to C{__init__}.
        """
        return self._address