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