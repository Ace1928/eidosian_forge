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
class TunnelAddressTests(SynchronousTestCase):
    """
    Tests for L{TunnelAddress}.
    """

    def test_interfaces(self):
        """
        A L{TunnelAddress} instances provides L{IAddress}.
        """
        self.assertTrue(verifyObject(IAddress, TunnelAddress(TunnelFlags.IFF_TAP, 'tap0')))

    def test_indexing(self):
        """
        A L{TunnelAddress} instance can be indexed to retrieve either the byte
        string C{"TUNTAP"} or the name of the tunnel interface, while
        triggering a deprecation warning.
        """
        address = TunnelAddress(TunnelFlags.IFF_TAP, 'tap0')
        self.assertEqual('TUNTAP', address[0])
        self.assertEqual('tap0', address[1])
        warnings = self.flushWarnings([self.test_indexing])
        message = 'TunnelAddress.__getitem__ is deprecated since Twisted 14.0.0  Use attributes instead.'
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])
        self.assertEqual(DeprecationWarning, warnings[1]['category'])
        self.assertEqual(message, warnings[1]['message'])
        self.assertEqual(2, len(warnings))

    def test_repr(self):
        """
        The string representation of a L{TunnelAddress} instance includes the
        class name and the values of the C{type} and C{name} attributes.
        """
        self.assertRegex(repr(TunnelAddress(TunnelFlags.IFF_TUN, name=b'device')), "TunnelAddress type=IFF_TUN name=b'device'>")