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
def toBox(self, name, strings, objects, proto):
    """
        Populate an 'out' AmpBox with strings encoded from an 'in' dictionary
        mapping names to Python values.

        @param name: the argument name to retrieve
        @type name: C{bytes}

        @param strings: The AmpBox to write string(s) to, a mapping of
        argument names to string values.
        @type strings: AmpBox

        @param objects: The dictionary to read object(s) from, a mapping of
        names to Python objects. Keys should be native strings.

        @type objects: dict

        @param proto: the protocol we are converting for.
        @type proto: AMP
        """
    obj = self.retrieve(objects, _wireNameToPythonIdentifier(name), proto)
    if self.optional and obj is None:
        pass
    else:
        strings[name] = self.toStringProto(obj, proto)