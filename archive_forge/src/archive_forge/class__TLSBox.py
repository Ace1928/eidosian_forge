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
class _TLSBox(AmpBox):
    """
    I am an AmpBox that, upon being sent, initiates a TLS connection.
    """
    __slots__: List[str] = []

    def __init__(self):
        if ssl is None:
            raise RemoteAmpError(b'TLS_ERROR', 'TLS not available')
        AmpBox.__init__(self)

    @property
    def certificate(self):
        return self.get(b'tls_localCertificate', _NoCertificate(False))

    @property
    def verify(self):
        return self.get(b'tls_verifyAuthorities', None)

    def _sendTo(self, proto):
        """
        Send my encoded value to the protocol, then initiate TLS.
        """
        ab = AmpBox(self)
        for k in [b'tls_localCertificate', b'tls_verifyAuthorities']:
            ab.pop(k, None)
        ab._sendTo(proto)
        proto._startTLS(self.certificate, self.verify)