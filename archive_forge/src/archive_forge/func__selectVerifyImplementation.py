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
def _selectVerifyImplementation():
    """
    Determine if C{service_identity} is installed. If so, use it. If not, use
    simplistic and incorrect checking as implemented in
    L{simpleVerifyHostname}.

    @return: 2-tuple of (C{verify_hostname}, C{VerificationError})
    @rtype: L{tuple}
    """
    whatsWrong = 'Without the service_identity module, Twisted can perform only rudimentary TLS client hostname verification.  Many valid certificate/hostname mappings may be rejected.'
    try:
        from service_identity import VerificationError
        from service_identity.pyopenssl import verify_hostname, verify_ip_address
        return (verify_hostname, verify_ip_address, VerificationError)
    except ImportError as e:
        warnings.warn_explicit("You do not have a working installation of the service_identity module: '" + str(e) + "'.  Please install it from <https://pypi.python.org/pypi/service_identity> and make sure all of its dependencies are satisfied.  " + whatsWrong, category=UserWarning, filename='', lineno=0)
    return (simpleVerifyHostname, simpleVerifyIPAddress, SimpleVerificationError)