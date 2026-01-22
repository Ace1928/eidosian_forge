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
def _makeContext(self):
    ctx = self._contextFactory(self.method)
    ctx.set_options(self._options)
    ctx.set_mode(self._mode)
    if self.certificate is not None and self.privateKey is not None:
        ctx.use_certificate(self.certificate)
        ctx.use_privatekey(self.privateKey)
        for extraCert in self.extraCertChain:
            ctx.add_extra_chain_cert(extraCert)
        ctx.check_privatekey()
    verifyFlags = SSL.VERIFY_NONE
    if self.verify:
        verifyFlags = SSL.VERIFY_PEER
        if self.requireCertificate:
            verifyFlags |= SSL.VERIFY_FAIL_IF_NO_PEER_CERT
        if self.verifyOnce:
            verifyFlags |= SSL.VERIFY_CLIENT_ONCE
        self.trustRoot._addCACertsToContext(ctx)
    ctx.set_verify(verifyFlags)
    if self.verifyDepth is not None:
        ctx.set_verify_depth(self.verifyDepth)
    sessionIDContext = hexlify(secureRandom(7))
    ctx.set_session_id(sessionIDContext)
    if self.enableSessions:
        ctx.set_session_cache_mode(SSL.SESS_CACHE_SERVER)
    else:
        ctx.set_session_cache_mode(SSL.SESS_CACHE_OFF)
    if self.dhParameters:
        ctx.load_tmp_dh(self.dhParameters._dhFile.path)
    ctx.set_cipher_list(self._cipherString.encode('ascii'))
    self._ecChooser.configureECDHCurve(ctx)
    if self._acceptableProtocols:
        _setAcceptableProtocols(ctx, self._acceptableProtocols)
    return ctx