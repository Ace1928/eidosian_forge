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
@implementer(IFileDescriptorReceiver)
class _DescriptorExchanger:
    """
    L{_DescriptorExchanger} is a mixin for L{BinaryBoxProtocol} which adds
    support for receiving file descriptors, a feature offered by
    L{IUNIXTransport<twisted.internet.interfaces.IUNIXTransport>}.

    @ivar _descriptors: Temporary storage for all file descriptors received.
        Values in this dictionary are the file descriptors (as integers).  Keys
        in this dictionary are ordinals giving the order in which each
        descriptor was received.  The ordering information is used to allow
        L{Descriptor} to determine which is the correct descriptor for any
        particular usage of that argument type.
    @type _descriptors: C{dict}

    @ivar _sendingDescriptorCounter: A no-argument callable which returns the
        ordinals, starting from 0.  This is used to construct values for
        C{_sendFileDescriptor}.

    @ivar _receivingDescriptorCounter: A no-argument callable which returns the
        ordinals, starting from 0.  This is used to construct values for
        C{fileDescriptorReceived}.
    """

    def __init__(self):
        self._descriptors = {}
        self._getDescriptor = self._descriptors.pop
        self._sendingDescriptorCounter = partial(next, count())
        self._receivingDescriptorCounter = partial(next, count())

    def _sendFileDescriptor(self, descriptor):
        """
        Assign and return the next ordinal to the given descriptor after sending
        the descriptor over this protocol's transport.
        """
        self.transport.sendFileDescriptor(descriptor)
        return self._sendingDescriptorCounter()

    def fileDescriptorReceived(self, descriptor):
        """
        Collect received file descriptors to be claimed later by L{Descriptor}.

        @param descriptor: The received file descriptor.
        @type descriptor: C{int}
        """
        self._descriptors[self._receivingDescriptorCounter()] = descriptor