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
class TunnelFlags(Flags):
    """
    L{TunnelFlags} defines more flags which are used to configure the behavior
    of a tunnel device.

    @cvar IFF_TUN: This indicates a I{tun}-type device.  This type of tunnel
        carries IP datagrams.  This flag is mutually exclusive with C{IFF_TAP}.

    @cvar IFF_TAP: This indicates a I{tap}-type device.  This type of tunnel
        carries ethernet frames.  This flag is mutually exclusive with C{IFF_TUN}.

    @cvar IFF_NO_PI: This indicates the I{protocol information} header will
        B{not} be included in data read from the tunnel.

    @see: U{https://www.kernel.org/doc/Documentation/networking/tuntap.txt}
    """
    IFF_TUN = FlagConstant(1)
    IFF_TAP = FlagConstant(2)
    TUN_FASYNC = FlagConstant(16)
    TUN_NOCHECKSUM = FlagConstant(32)
    TUN_NO_PI = FlagConstant(64)
    TUN_ONE_QUEUE = FlagConstant(128)
    TUN_PERSIST = FlagConstant(256)
    TUN_VNET_HDR = FlagConstant(512)
    IFF_NO_PI = FlagConstant(4096)
    IFF_ONE_QUEUE = FlagConstant(8192)
    IFF_VNET_HDR = FlagConstant(16384)
    IFF_TUN_EXCL = FlagConstant(32768)