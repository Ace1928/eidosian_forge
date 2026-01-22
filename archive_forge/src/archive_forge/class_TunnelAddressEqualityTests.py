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
class TunnelAddressEqualityTests(SynchronousTestCase):
    """
    Tests for the implementation of equality (C{==} and C{!=}) for
    L{TunnelAddress}.
    """

    def setUp(self):
        self.first = TunnelAddress(TunnelFlags.IFF_TUN, b'device')
        self.second = TunnelAddress(TunnelFlags.IFF_TUN | TunnelFlags.IFF_TUN, b'device')
        self.variedType = TunnelAddress(TunnelFlags.IFF_TAP, b'tap1')
        self.variedName = TunnelAddress(TunnelFlags.IFF_TUN, b'tun1')

    def test_selfComparesEqual(self):
        """
        A L{TunnelAddress} compares equal to itself.
        """
        self.assertTrue(self.first == self.first)

    def test_selfNotComparesNotEqual(self):
        """
        A L{TunnelAddress} doesn't compare not equal to itself.
        """
        self.assertFalse(self.first != self.first)

    def test_sameAttributesComparesEqual(self):
        """
        Two L{TunnelAddress} instances with the same value for the C{type} and
        C{name} attributes compare equal to each other.
        """
        self.assertTrue(self.first == self.second)

    def test_sameAttributesNotComparesNotEqual(self):
        """
        Two L{TunnelAddress} instances with the same value for the C{type} and
        C{name} attributes don't compare not equal to each other.
        """
        self.assertFalse(self.first != self.second)

    def test_differentTypeComparesNotEqual(self):
        """
        Two L{TunnelAddress} instances that differ only by the value of their
        type don't compare equal to each other.
        """
        self.assertFalse(self.first == self.variedType)

    def test_differentTypeNotComparesEqual(self):
        """
        Two L{TunnelAddress} instances that differ only by the value of their
        type compare not equal to each other.
        """
        self.assertTrue(self.first != self.variedType)

    def test_differentNameComparesNotEqual(self):
        """
        Two L{TunnelAddress} instances that differ only by the value of their
        name don't compare equal to each other.
        """
        self.assertFalse(self.first == self.variedName)

    def test_differentNameNotComparesEqual(self):
        """
        Two L{TunnelAddress} instances that differ only by the value of their
        name compare not equal to each other.
        """
        self.assertTrue(self.first != self.variedName)

    def test_differentClassNotComparesEqual(self):
        """
        A L{TunnelAddress} doesn't compare equal to an instance of another
        class.
        """
        self.assertFalse(self.first == self)

    def test_differentClassComparesNotEqual(self):
        """
        A L{TunnelAddress} compares not equal to an instance of another class.
        """
        self.assertTrue(self.first != self)