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
@lru_cache(maxsize=32)
def _expandCipherString(cipherString, method, options):
    """
    Expand C{cipherString} according to C{method} and C{options} to a tuple of
    explicit ciphers that are supported by the current platform.

    @param cipherString: An OpenSSL cipher string to expand.
    @type cipherString: L{unicode}

    @param method: An OpenSSL method like C{SSL.TLS_METHOD} used for
        determining the effective ciphers.

    @param options: OpenSSL options like C{SSL.OP_NO_SSLv3} ORed together.
    @type options: L{int}

    @return: The effective list of explicit ciphers that results from the
        arguments on the current platform.
    @rtype: L{tuple} of L{ICipher}
    """
    ctx = SSL.Context(method)
    ctx.set_options(options)
    try:
        ctx.set_cipher_list(cipherString.encode('ascii'))
    except SSL.Error as e:
        if not e.args[0]:
            return tuple()
        if e.args[0][0][2] == 'no cipher match':
            return tuple()
        else:
            raise
    conn = SSL.Connection(ctx, None)
    ciphers = conn.get_cipher_list()
    if isinstance(ciphers[0], str):
        return tuple((OpenSSLCipher(cipher) for cipher in ciphers))
    else:
        return tuple((OpenSSLCipher(cipher.decode('ascii')) for cipher in ciphers))