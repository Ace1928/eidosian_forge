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
class IOpenSSLTrustRoot(Interface):
    """
    Trust settings for an OpenSSL context.

    Note that this interface's methods are private, so things outside of
    Twisted shouldn't implement it.
    """

    def _addCACertsToContext(context):
        """
        Add certificate-authority certificates to an SSL context whose
        connections should trust those authorities.

        @param context: An SSL context for a connection which should be
            verified by some certificate authority.
        @type context: L{OpenSSL.SSL.Context}

        @return: L{None}
        """