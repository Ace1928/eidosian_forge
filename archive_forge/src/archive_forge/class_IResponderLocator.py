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
class IResponderLocator(Interface):
    """
    An application object which can look up appropriate responder methods for
    AMP commands.
    """

    def locateResponder(name):
        """
        Locate a responder method appropriate for the named command.

        @param name: the wire-level name (commandName) of the AMP command to be
        responded to.
        @type name: C{bytes}

        @return: a 1-argument callable that takes an L{AmpBox} with argument
        values for the given command, and returns an L{AmpBox} containing
        argument values for the named command, or a L{Deferred} that fires the
        same.
        """