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
def _errorReceived(self, box):
    """
        An AMP box was received that answered a command previously sent with
        L{callRemote}, with an error.

        @param box: an L{AmpBox} with a value for its L{ERROR}, L{ERROR_CODE},
        and L{ERROR_DESCRIPTION} keys.
        """
    question = self._outstandingRequests.pop(box[ERROR])
    question.addErrback(self.unhandledError)
    errorCode = box[ERROR_CODE]
    description = box[ERROR_DESCRIPTION]
    if isinstance(description, bytes):
        description = description.decode('utf-8', 'replace')
    if errorCode in PROTOCOL_ERRORS:
        exc = PROTOCOL_ERRORS[errorCode](errorCode, description)
    else:
        exc = RemoteAmpError(errorCode, description)
    question.errback(Failure(exc))