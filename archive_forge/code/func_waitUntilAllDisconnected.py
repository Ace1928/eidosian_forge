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
def waitUntilAllDisconnected(reactor, protocols):
    """
    Take a list of disconnecting protocols, callback a L{Deferred} when they're
    all done.

    This is a hack to make some older tests less flaky, as
    L{ITransport.loseConnection} is not atomic on all reactors (for example,
    the CoreFoundation, which sometimes takes a reactor turn for CFSocket to
    realise). New tests should either not use real sockets in testing, or take
    the advice in
    I{https://jml.io/pages/how-to-disconnect-in-twisted-really.html} to heart.

    @param reactor: The reactor to schedule the checks on.
    @type reactor: L{IReactorTime}

    @param protocols: The protocols to wait for disconnecting.
    @type protocols: A L{list} of L{IProtocol}s.
    """
    lc = None

    def _check():
        if True not in [x.transport.connected for x in protocols]:
            lc.stop()
    lc = task.LoopingCall(_check)
    lc.clock = reactor
    return lc.start(0.01, now=True)