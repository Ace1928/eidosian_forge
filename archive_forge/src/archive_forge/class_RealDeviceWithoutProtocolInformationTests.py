import os
import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINVAL, ENODEV, ENOENT, EPERM, EWOULDBLOCK
from itertools import cycle
from random import randrange
from signal import SIGINT
from typing import Optional
from twisted.python.reflect import ObjectNotFound, namedAny
from zope.interface import Interface, implementer
from zope.interface.verify import verifyObject
from twisted.internet.error import CannotListenError
from twisted.internet.interfaces import IAddress, IListeningPort, IReactorFDSet
from twisted.internet.protocol import (
from twisted.internet.task import Clock
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.raw import IRawPacketProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.python.compat import iterbytes
from twisted.python.log import addObserver, removeObserver, textFromEventDict
from twisted.python.reflect import fullyQualifiedName
from twisted.trial.unittest import SkipTest, SynchronousTestCase
class RealDeviceWithoutProtocolInformationTests(RealDeviceTestsMixin, TunnelDeviceTestsMixin, SynchronousTestCase):
    """
    Run various tap-type tunnel unit tests, with "protocol information" (PI)
    turned off, against a real I/O system.
    """
    _TUNNEL_DEVICE = b'tap-twtest'
    _TUNNEL_LOCAL = b'172.16.0.1'
    _TUNNEL_REMOTE = b'172.16.0.2'
    helper = TapHelper(_TUNNEL_REMOTE, _TUNNEL_LOCAL, pi=False)