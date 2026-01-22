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
def _setAcceptableProtocols(context, acceptableProtocols):
    """
    Called to set up the L{OpenSSL.SSL.Context} for doing NPN and/or ALPN
    negotiation.

    @param context: The context which is set up.
    @type context: L{OpenSSL.SSL.Context}

    @param acceptableProtocols: The protocols this peer is willing to speak
        after the TLS negotiation has completed, advertised over both ALPN and
        NPN. If this argument is specified, and no overlap can be found with
        the other peer, the connection will fail to be established. If the
        remote peer does not offer NPN or ALPN, the connection will be
        established, but no protocol wil be negotiated. Protocols earlier in
        the list are preferred over those later in the list.
    @type acceptableProtocols: L{list} of L{bytes}
    """

    def protoSelectCallback(conn, protocols):
        """
        NPN client-side and ALPN server-side callback used to select
        the next protocol. Prefers protocols found earlier in
        C{_acceptableProtocols}.

        @param conn: The context which is set up.
        @type conn: L{OpenSSL.SSL.Connection}

        @param conn: Protocols advertised by the other side.
        @type conn: L{list} of L{bytes}
        """
        overlap = set(protocols) & set(acceptableProtocols)
        for p in acceptableProtocols:
            if p in overlap:
                return p
        else:
            return b''
    if not acceptableProtocols:
        return
    supported = protocolNegotiationMechanisms()
    if supported & ProtocolNegotiationSupport.NPN:

        def npnAdvertiseCallback(conn):
            return acceptableProtocols
        context.set_npn_advertise_callback(npnAdvertiseCallback)
        context.set_npn_select_callback(protoSelectCallback)
    if supported & ProtocolNegotiationSupport.ALPN:
        context.set_alpn_select_callback(protoSelectCallback)
        context.set_alpn_protos(acceptableProtocols)