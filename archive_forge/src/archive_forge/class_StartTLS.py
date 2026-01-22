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
class StartTLS(Command):
    """
    Use, or subclass, me to implement a command that starts TLS.

    Callers of StartTLS may pass several special arguments, which affect the
    TLS negotiation:

        - tls_localCertificate: This is a
        twisted.internet.ssl.PrivateCertificate which will be used to secure
        the side of the connection it is returned on.

        - tls_verifyAuthorities: This is a list of
        twisted.internet.ssl.Certificate objects that will be used as the
        certificate authorities to verify our peer's certificate.

    Each of those special parameters may also be present as a key in the
    response dictionary.
    """
    arguments = [(b'tls_localCertificate', _LocalArgument(optional=True)), (b'tls_verifyAuthorities', _LocalArgument(optional=True))]
    response = [(b'tls_localCertificate', _LocalArgument(optional=True)), (b'tls_verifyAuthorities', _LocalArgument(optional=True))]
    responseType = _TLSBox

    def __init__(self, *, tls_localCertificate=None, tls_verifyAuthorities=None, **kw):
        """
        Create a StartTLS command.  (This is private.  Use AMP.callRemote.)

        @param tls_localCertificate: the PrivateCertificate object to use to
        secure the connection.  If it's L{None}, or unspecified, an ephemeral DH
        key is used instead.

        @param tls_verifyAuthorities: a list of Certificate objects which
        represent root certificates to verify our peer with.
        """
        if ssl is None:
            raise RuntimeError('TLS not available.')
        self.certificate = _NoCertificate(True) if tls_localCertificate is None else tls_localCertificate
        self.authorities = tls_verifyAuthorities
        Command.__init__(self, **kw)

    def _doCommand(self, proto):
        """
        When a StartTLS command is sent, prepare to start TLS, but don't actually
        do it; wait for the acknowledgement, then initiate the TLS handshake.
        """
        d = Command._doCommand(self, proto)
        proto._prepareTLS(self.certificate, self.authorities)

        def actuallystart(response):
            proto._startTLS(self.certificate, self.authorities)
            return response
        d.addCallback(actuallystart)
        return d