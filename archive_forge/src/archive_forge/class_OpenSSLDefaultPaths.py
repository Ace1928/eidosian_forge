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
@implementer(IOpenSSLTrustRoot)
class OpenSSLDefaultPaths:
    """
    Trust the set of default verify paths that OpenSSL was built with, as
    specified by U{SSL_CTX_set_default_verify_paths
    <https://www.openssl.org/docs/man1.1.1/man3/SSL_CTX_load_verify_locations.html>}.
    """

    def _addCACertsToContext(self, context):
        context.set_default_verify_paths()