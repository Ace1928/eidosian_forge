import struct
import sys
from io import BytesIO
from typing import List, Optional, Type
from zope.interface.verify import verifyObject
from twisted.internet import protocol, task
from twisted.internet.interfaces import IProducer
from twisted.internet.protocol import connectionDone
from twisted.protocols import basic
from twisted.python.compat import iterbytes
from twisted.python.failure import Failure
from twisted.test import proto_helpers
from twisted.trial import unittest
class TestInt32(TestMixin, basic.Int32StringReceiver):
    """
    A L{basic.Int32StringReceiver} storing received strings in an array.

    @ivar received: array holding received strings.
    """