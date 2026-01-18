import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_toStringProto(self):
    """
        To send a file descriptor, L{Descriptor.toStringProto} uses the
        L{IUNIXTransport.sendFileDescriptor} implementation of the transport of
        the protocol passed to it to copy the file descriptor.  Each subsequent
        descriptor sent over a particular AMP connection is assigned the next
        integer value, starting from 0.  The base ten string representation of
        this value is the byte encoding of the argument.

        This is a whitebox test which involves direct L{_DescriptorExchanger}
        state inspection and mutation.
        """
    argument = amp.Descriptor()
    self.assertEqual(b'0', argument.toStringProto(2, self.protocol))
    self.assertEqual(('fileDescriptorReceived', 2 + self.fuzz), self.transport._queue.pop(0))
    self.assertEqual(b'1', argument.toStringProto(4, self.protocol))
    self.assertEqual(('fileDescriptorReceived', 4 + self.fuzz), self.transport._queue.pop(0))
    self.assertEqual(b'2', argument.toStringProto(6, self.protocol))
    self.assertEqual(('fileDescriptorReceived', 6 + self.fuzz), self.transport._queue.pop(0))
    self.assertEqual({}, self.protocol._descriptors)