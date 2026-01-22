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
class _TunnelDescription(namedtuple('_TunnelDescription', 'fileno name')):
    """
    Describe an existing tunnel.

    @ivar fileno: the file descriptor associated with the tunnel
    @type fileno: L{int}

    @ivar name: the name of the tunnel
    @type name: L{bytes}
    """