from __future__ import annotations
import warnings
from binascii import hexlify
from functools import lru_cache
from hashlib import md5
from typing import Dict
from zope.interface import Interface, implementer
from OpenSSL import SSL, crypto
from OpenSSL._util import lib as pyOpenSSLlib
import attr
from constantly import FlagConstant, Flags, NamedConstant, Names
from incremental import Version
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import CertificateError, VerifyError
from twisted.internet.interfaces import (
from twisted.python import log, util
from twisted.python.compat import nativeString
from twisted.python.deprecate import _mutuallyExclusiveArguments, deprecated
from twisted.python.failure import Failure
from twisted.python.randbytes import secureRandom
from ._idna import _idnaBytes
@implementer(IOpenSSLClientConnectionCreator)
class ClientTLSOptions:
    """
    Client creator for TLS.

    Private implementation type (not exposed to applications) for public
    L{optionsForClientTLS} API.

    @ivar _ctx: The context to use for new connections.
    @type _ctx: L{OpenSSL.SSL.Context}

    @ivar _hostname: The hostname to verify, as specified by the application,
        as some human-readable text.
    @type _hostname: L{unicode}

    @ivar _hostnameBytes: The hostname to verify, decoded into IDNA-encoded
        bytes.  This is passed to APIs which think that hostnames are bytes,
        such as OpenSSL's SNI implementation.
    @type _hostnameBytes: L{bytes}

    @ivar _hostnameASCII: The hostname, as transcoded into IDNA ASCII-range
        unicode code points.  This is pre-transcoded because the
        C{service_identity} package is rather strict about requiring the
        C{idna} package from PyPI for internationalized domain names, rather
        than working with Python's built-in (but sometimes broken) IDNA
        encoding.  ASCII values, however, will always work.
    @type _hostnameASCII: L{unicode}

    @ivar _hostnameIsDnsName: Whether or not the C{_hostname} is a DNSName.
        Will be L{False} if C{_hostname} is an IP address or L{True} if
        C{_hostname} is a DNSName
    @type _hostnameIsDnsName: L{bool}
    """

    def __init__(self, hostname, ctx):
        """
        Initialize L{ClientTLSOptions}.

        @param hostname: The hostname to verify as input by a human.
        @type hostname: L{unicode}

        @param ctx: an L{OpenSSL.SSL.Context} to use for new connections.
        @type ctx: L{OpenSSL.SSL.Context}.
        """
        self._ctx = ctx
        self._hostname = hostname
        if isIPAddress(hostname) or isIPv6Address(hostname):
            self._hostnameBytes = hostname.encode('ascii')
            self._hostnameIsDnsName = False
        else:
            self._hostnameBytes = _idnaBytes(hostname)
            self._hostnameIsDnsName = True
        self._hostnameASCII = self._hostnameBytes.decode('ascii')
        ctx.set_info_callback(_tolerateErrors(self._identityVerifyingInfoCallback))

    def clientConnectionForTLS(self, tlsProtocol):
        """
        Create a TLS connection for a client.

        @note: This will call C{set_app_data} on its connection.  If you're
            delegating to this implementation of this method, don't ever call
            C{set_app_data} or C{set_info_callback} on the returned connection,
            or you'll break the implementation of various features of this
            class.

        @param tlsProtocol: the TLS protocol initiating the connection.
        @type tlsProtocol: L{twisted.protocols.tls.TLSMemoryBIOProtocol}

        @return: the configured client connection.
        @rtype: L{OpenSSL.SSL.Connection}
        """
        context = self._ctx
        connection = SSL.Connection(context, None)
        connection.set_app_data(tlsProtocol)
        return connection

    def _identityVerifyingInfoCallback(self, connection, where, ret):
        """
        U{info_callback
        <http://pythonhosted.org/pyOpenSSL/api/ssl.html#OpenSSL.SSL.Context.set_info_callback>
        } for pyOpenSSL that verifies the hostname in the presented certificate
        matches the one passed to this L{ClientTLSOptions}.

        @param connection: the connection which is handshaking.
        @type connection: L{OpenSSL.SSL.Connection}

        @param where: flags indicating progress through a TLS handshake.
        @type where: L{int}

        @param ret: ignored
        @type ret: ignored
        """
        if where & SSL.SSL_CB_HANDSHAKE_START and self._hostnameIsDnsName:
            connection.set_tlsext_host_name(self._hostnameBytes)
        elif where & SSL.SSL_CB_HANDSHAKE_DONE:
            try:
                if self._hostnameIsDnsName:
                    verifyHostname(connection, self._hostnameASCII)
                else:
                    verifyIPAddress(connection, self._hostnameASCII)
            except VerificationError:
                f = Failure()
                transport = connection.get_app_data()
                transport.failVerification(f)