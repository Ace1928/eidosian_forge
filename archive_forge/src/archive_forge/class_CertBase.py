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
class CertBase:
    """
    Base class for public (certificate only) and private (certificate + key
    pair) certificates.

    @ivar original: The underlying OpenSSL certificate object.
    @type original: L{OpenSSL.crypto.X509}
    """

    def __init__(self, original):
        self.original = original

    def _copyName(self, suffix):
        dn = DistinguishedName()
        dn._copyFrom(getattr(self.original, 'get_' + suffix)())
        return dn

    def getSubject(self):
        """
        Retrieve the subject of this certificate.

        @return: A copy of the subject of this certificate.
        @rtype: L{DistinguishedName}
        """
        return self._copyName('subject')

    def __conform__(self, interface):
        """
        Convert this L{CertBase} into a provider of the given interface.

        @param interface: The interface to conform to.
        @type interface: L{zope.interface.interfaces.IInterface}

        @return: an L{IOpenSSLTrustRoot} provider or L{NotImplemented}
        @rtype: L{IOpenSSLTrustRoot} or L{NotImplemented}
        """
        if interface is IOpenSSLTrustRoot:
            return OpenSSLCertificateAuthorities([self.original])
        return NotImplemented