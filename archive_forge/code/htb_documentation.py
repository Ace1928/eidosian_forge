from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp

        Make a C{Protocol} instance with a shaped transport.

        Any parameters will be passed on to the protocol's initializer.

        @returns: A C{Protocol} instance with a L{ShapedTransport}.
        