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
class BadLocalReturn(AmpError):
    """
    A bad value was returned from a local command; we were unable to coerce it.
    """

    def __init__(self, message: str, enclosed: Failure) -> None:
        AmpError.__init__(self)
        self.message = message
        self.enclosed = enclosed

    def __repr__(self) -> str:
        return self.message + ' ' + self.enclosed.getBriefTraceback()
    __str__ = __repr__