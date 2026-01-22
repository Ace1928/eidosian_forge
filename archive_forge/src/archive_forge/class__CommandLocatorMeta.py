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
class _CommandLocatorMeta(type):
    """
    This metaclass keeps track of all of the Command.responder-decorated
    methods defined since the last CommandLocator subclass was defined.  It
    assumes (usually correctly, but unfortunately not necessarily so) that
    those commands responders were all declared as methods of the class
    being defined.  Note that this list can be incorrect if users use the
    Command.responder decorator outside the context of a CommandLocator
    class declaration.

    Command responders defined on subclasses are given precedence over
    those inherited from a base class.

    The Command.responder decorator explicitly cooperates with this
    metaclass.
    """
    _currentClassCommands: 'list[tuple[type[Command], Callable[..., Any]]]' = []

    def __new__(cls, name, bases, attrs):
        commands = cls._currentClassCommands[:]
        cls._currentClassCommands[:] = []
        cd = attrs['_commandDispatch'] = {}
        subcls = type.__new__(cls, name, bases, attrs)
        ancestors = list(subcls.__mro__[1:])
        ancestors.reverse()
        for ancestor in ancestors:
            cd.update(getattr(ancestor, '_commandDispatch', {}))
        for commandClass, responderFunc in commands:
            cd[commandClass.commandName] = (commandClass, responderFunc)
        if bases and subcls.lookupFunction != CommandLocator.lookupFunction:

            def locateResponder(self, name):
                warnings.warn('Override locateResponder, not lookupFunction.', category=PendingDeprecationWarning, stacklevel=2)
                return self.lookupFunction(name)
            subcls.locateResponder = locateResponder
        return subcls