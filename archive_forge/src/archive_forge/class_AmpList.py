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
class AmpList(Argument):
    """
    Convert a list of dictionaries into a list of AMP boxes on the wire.

    For example, if you want to pass::

        [{'a': 7, 'b': u'hello'}, {'a': 9, 'b': u'goodbye'}]

    You might use an AmpList like this in your arguments or response list::

        AmpList([('a', Integer()),
                 ('b', Unicode())])
    """

    def __init__(self, subargs, optional=False):
        """
        Create an AmpList.

        @param subargs: a list of 2-tuples of ('name', argument) describing the
        schema of the dictionaries in the sequence of amp boxes.
        @type subargs: A C{list} of (C{bytes}, L{Argument}) tuples.

        @param optional: a boolean indicating whether this argument can be
        omitted in the protocol.
        """
        assert all((isinstance(name, bytes) for name, _ in subargs)), "AmpList should be defined with a list of (name, argument) tuples where `name' is a byte string, got: %r" % (subargs,)
        self.subargs = subargs
        Argument.__init__(self, optional)

    def fromStringProto(self, inString, proto):
        boxes = parseString(inString)
        values = [_stringsToObjects(box, self.subargs, proto) for box in boxes]
        return values

    def toStringProto(self, inObject, proto):
        return b''.join([_objectsToStrings(objects, self.subargs, Box(), proto).serialize() for objects in inObject])