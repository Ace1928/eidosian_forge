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
def _sendBoxCommand(self, command, box, requiresAnswer=True):
    """
        Send a command across the wire with the given C{amp.Box}.

        Mutate the given box to give it any additional keys (_command, _ask)
        required for the command and request/response machinery, then send it.

        If requiresAnswer is True, returns a C{Deferred} which fires when a
        response is received. The C{Deferred} is fired with an C{amp.Box} on
        success, or with an C{amp.RemoteAmpError} if an error is received.

        If the Deferred fails and the error is not handled by the caller of
        this method, the failure will be logged and the connection dropped.

        @param command: a C{bytes}, the name of the command to issue.

        @param box: an AmpBox with the arguments for the command.

        @param requiresAnswer: a boolean.  Defaults to True.  If True, return a
        Deferred which will fire when the other side responds to this command.
        If False, return None and do not ask the other side for acknowledgement.

        @return: a Deferred which fires the AmpBox that holds the response to
        this command, or None, as specified by requiresAnswer.

        @raise ProtocolSwitched: if the protocol has been switched.
        """
    if self._failAllReason is not None:
        if requiresAnswer:
            return fail(self._failAllReason)
        else:
            return None
    box[COMMAND] = command
    tag = self._nextTag()
    if requiresAnswer:
        box[ASK] = tag
    box._sendTo(self.boxSender)
    if requiresAnswer:
        result = self._outstandingRequests[tag] = Deferred()
    else:
        result = None
    return result