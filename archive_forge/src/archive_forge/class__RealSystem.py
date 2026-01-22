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
class _RealSystem:
    """
    An interface to the parts of the operating system which L{TuntapPort}
    relies on.  This is most of an implementation of L{_IInputOutputSystem}.
    """
    open = staticmethod(os.open)
    read = staticmethod(os.read)
    write = staticmethod(os.write)
    close = staticmethod(os.close)
    ioctl = staticmethod(fcntl.ioctl)
    O_RDWR = os.O_RDWR
    O_NONBLOCK = os.O_NONBLOCK
    O_CLOEXEC = getattr(os, 'O_CLOEXEC', 524288)