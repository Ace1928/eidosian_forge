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
@implementer(interfaces.IAddress)
class TunnelAddress(FancyStrMixin, FancyEqMixin):
    """
    A L{TunnelAddress} represents the tunnel to which a L{TuntapPort} is bound.
    """
    compareAttributes = ('_typeValue', 'name')
    showAttributes = (('type', lambda flag: flag.name), 'name')

    @property
    def _typeValue(self):
        """
        Return the integer value of the C{type} attribute.  Used to produce
        correct results in the equality implementation.
        """
        return self.type.value

    def __init__(self, type, name):
        """
        @param type: Either L{TunnelFlags.IFF_TUN} or L{TunnelFlags.IFF_TAP},
            representing the type of this tunnel.

        @param name: The system name of the tunnel.
        @type name: L{bytes}
        """
        self.type = type
        self.name = name

    def __getitem__(self, index):
        """
        Deprecated accessor for the tunnel name.  Use attributes instead.
        """
        warnings.warn('TunnelAddress.__getitem__ is deprecated since Twisted 14.0.0  Use attributes instead.', category=DeprecationWarning, stacklevel=2)
        return ('TUNTAP', self.name)[index]