from __future__ import annotations
import datetime
import decimal
import warnings
from functools import partial
from io import BytesIO
from itertools import count
from struct import pack
from types import MethodType
from typing import (
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, fail, maybeDeferred
from twisted.internet.error import ConnectionClosed, ConnectionLost, PeerVerifyError
from twisted.internet.interfaces import IFileDescriptorReceiver
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.basic import Int16StringReceiver, StatefulStringProtocol
from twisted.python import filepath, log
from twisted.python._tzhelper import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.reflect import accumulateClassDict
def toStringProto(self, inObject, proto):
    """
        Send C{inObject}, an integer file descriptor, over C{proto}'s connection
        and return a unique identifier which will allow the receiver to
        associate the file descriptor with this argument.

        @param inObject: A file descriptor to duplicate over an AMP connection
            as the value for this argument.
        @type inObject: C{int}

        @param proto: The protocol which will be used to send this descriptor.
            This protocol must be connected via a transport providing
            L{IUNIXTransport<twisted.internet.interfaces.IUNIXTransport>}.

        @return: A byte string which can be used by the receiver to reconstruct
            the file descriptor.
        @rtype: C{bytes}
        """
    identifier = proto._sendFileDescriptor(inObject)
    outString = Integer.toStringProto(self, identifier, proto)
    return outString