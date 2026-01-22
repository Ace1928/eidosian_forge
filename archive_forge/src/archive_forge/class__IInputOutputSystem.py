import errno
import fcntl
import os
import platform
import struct
import warnings
from collections import namedtuple
from typing import Tuple
from zope.interface import Attribute, Interface, implementer
from constantly import FlagConstant, Flags
from incremental import Version
from twisted.internet import abstract, defer, error, interfaces, task
from twisted.pair import ethernet, raw
from twisted.python import log
from twisted.python.deprecate import deprecated
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import FancyEqMixin, FancyStrMixin
class _IInputOutputSystem(Interface):
    """
    An interface for performing some basic kinds of I/O (particularly that I/O
    which might be useful for L{twisted.pair.tuntap}-using code).
    """
    O_RDWR = Attribute('@see: L{os.O_RDWR}')
    O_NONBLOCK = Attribute('@see: L{os.O_NONBLOCK}')
    O_CLOEXEC = Attribute('@see: L{os.O_CLOEXEC}')

    def open(filename, flag, mode=511):
        """
        @see: L{os.open}
        """

    def ioctl(fd, opt, arg=None, mutate_flag=None):
        """
        @see: L{fcntl.ioctl}
        """

    def read(fd, limit):
        """
        @see: L{os.read}
        """

    def write(fd, data):
        """
        @see: L{os.write}
        """

    def close(fd):
        """
        @see: L{os.close}
        """

    def sendUDP(datagram, address):
        """
        Send a datagram to a certain address.

        @param datagram: The payload of a UDP datagram to send.
        @type datagram: L{bytes}

        @param address: The destination to which to send the datagram.
        @type address: L{tuple} of (L{bytes}, L{int})

        @return: The local address from which the datagram was sent.
        @rtype: L{tuple} of (L{bytes}, L{int})
        """

    def receiveUDP(fileno, host, port):
        """
        Return a socket which can be used to receive datagrams sent to the
        given address.

        @param fileno: A file descriptor representing a tunnel device which the
            datagram was either sent via or will be received via.
        @type fileno: L{int}

        @param host: The IPv4 address at which the datagram will be received.
        @type host: L{bytes}

        @param port: The UDP port number at which the datagram will be
            received.
        @type port: L{int}

        @return: A L{socket.socket} which can be used to receive the specified
            datagram.
        """