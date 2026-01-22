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
class _CommandMeta(type):
    """
    Metaclass hack to establish reverse-mappings for 'errors' and
    'fatalErrors' as class vars.
    """

    def __new__(cls: type[_Self], name: str, bases: tuple[type], attrs: dict[str, object]) -> Type[Command]:
        reverseErrors = attrs['reverseErrors'] = {}
        er = attrs['allErrors'] = {}
        if 'commandName' not in attrs:
            attrs['commandName'] = name.encode('ascii')
        newtype: Type[Command] = type.__new__(cls, name, bases, attrs)
        if not isinstance(newtype.commandName, bytes):
            raise TypeError('Command names must be byte strings, got: {!r}'.format(newtype.commandName))
        for bname, _ in newtype.arguments:
            if not isinstance(bname, bytes):
                raise TypeError(f'Argument names must be byte strings, got: {bname!r}')
        for bname, _ in newtype.response:
            if not isinstance(bname, bytes):
                raise TypeError(f'Response names must be byte strings, got: {bname!r}')
        errors: Dict[Type[Exception], bytes] = {}
        fatalErrors: Dict[Type[Exception], bytes] = {}
        accumulateClassDict(newtype, 'errors', errors)
        accumulateClassDict(newtype, 'fatalErrors', fatalErrors)
        if not isinstance(newtype.errors, dict):
            newtype.errors = dict(newtype.errors)
        if not isinstance(newtype.fatalErrors, dict):
            newtype.fatalErrors = dict(newtype.fatalErrors)
        for v, k in errors.items():
            reverseErrors[k] = v
            er[v] = k
        for v, k in fatalErrors.items():
            reverseErrors[k] = v
            er[v] = k
        for _, bname in newtype.errors.items():
            if not isinstance(bname, bytes):
                raise TypeError(f'Error names must be byte strings, got: {bname!r}')
        for _, bname in newtype.fatalErrors.items():
            if not isinstance(bname, bytes):
                raise TypeError(f'Fatal error names must be byte strings, got: {bname!r}')
        return newtype