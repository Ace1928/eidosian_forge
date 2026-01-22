from __future__ import annotations
from typing import Callable, Iterable, Optional, cast
from zope.interface import directlyProvides, implementer, providedBy
from OpenSSL.SSL import Connection, Error, SysCallError, WantReadError, ZeroReturnError
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.policies import ProtocolWrapper, WrappingFactory
from twisted.python.failure import Failure
def _representsEOF(exceptionObject: Error) -> bool:
    """
    Does the given OpenSSL.SSL.Error represent an end-of-file?
    """
    reasonString: str
    if isinstance(exceptionObject, SysCallError):
        _, reasonString = exceptionObject.args
    else:
        errorQueue = exceptionObject.args[0]
        _, _, reasonString = errorQueue[-1]
    return reasonString.casefold().startswith('unexpected eof')