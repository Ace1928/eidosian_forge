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
@implementer(IOpenSSLContextFactory)
class OpenSSLCertificateOptions:
    """
    A L{CertificateOptions <twisted.internet.ssl.CertificateOptions>} specifies
    the security properties for a client or server TLS connection used with
    OpenSSL.

    @ivar _options: Any option flags to set on the L{OpenSSL.SSL.Context}
        object that will be created.
    @type _options: L{int}

    @ivar _cipherString: An OpenSSL-specific cipher string.
    @type _cipherString: L{unicode}

    @ivar _defaultMinimumTLSVersion: The default TLS version that will be
        negotiated.  This should be a "safe default", with wide client and
        server support, vs an optimally secure one that excludes a large number
        of users.  As of May 2022, TLSv1.2 is that safe default.
    @type _defaultMinimumTLSVersion: L{TLSVersion} constant
    """
    _contextFactory = SSL.Context
    _context = None
    _OP_NO_TLSv1_3 = _tlsDisableFlags[TLSVersion.TLSv1_3]
    _defaultMinimumTLSVersion = TLSVersion.TLSv1_2

    @_mutuallyExclusiveArguments([['trustRoot', 'requireCertificate'], ['trustRoot', 'verify'], ['trustRoot', 'caCerts'], ['method', 'insecurelyLowerMinimumTo'], ['method', 'raiseMinimumTo'], ['raiseMinimumTo', 'insecurelyLowerMinimumTo'], ['method', 'lowerMaximumSecurityTo']])
    def __init__(self, privateKey=None, certificate=None, method=None, verify=False, caCerts=None, verifyDepth=9, requireCertificate=True, verifyOnce=True, enableSingleUseKeys=True, enableSessions=False, fixBrokenPeers=False, enableSessionTickets=False, extraCertChain=None, acceptableCiphers=None, dhParameters=None, trustRoot=None, acceptableProtocols=None, raiseMinimumTo=None, insecurelyLowerMinimumTo=None, lowerMaximumSecurityTo=None):
        """
        Create an OpenSSL context SSL connection context factory.

        @param privateKey: A PKey object holding the private key.

        @param certificate: An X509 object holding the certificate.

        @param method: Deprecated, use a combination of
            C{insecurelyLowerMinimumTo}, C{raiseMinimumTo}, or
            C{lowerMaximumSecurityTo} instead.  The SSL protocol to use, one of
            C{TLS_METHOD}, C{TLSv1_2_METHOD}, or C{TLSv1_2_METHOD} (or any
            future method constants provided by pyOpenSSL).  By default, a
            setting will be used which allows TLSv1.2 and TLSv1.3.  Can not be
            used with C{insecurelyLowerMinimumTo}, C{raiseMinimumTo}, or
            C{lowerMaximumSecurityTo}.

        @param verify: Please use a C{trustRoot} keyword argument instead,
            since it provides the same functionality in a less error-prone way.
            By default this is L{False}.

            If L{True}, verify certificates received from the peer and fail the
            handshake if verification fails.  Otherwise, allow anonymous
            sessions and sessions with certificates which fail validation.

        @param caCerts: Please use a C{trustRoot} keyword argument instead,
            since it provides the same functionality in a less error-prone way.

            List of certificate authority certificate objects to use to verify
            the peer's certificate.  Only used if verify is L{True} and will be
            ignored otherwise.  Since verify is L{False} by default, this is
            L{None} by default.

        @type caCerts: L{list} of L{OpenSSL.crypto.X509}

        @param verifyDepth: Depth in certificate chain down to which to verify.
            If unspecified, use the underlying default (9).

        @param requireCertificate: Please use a C{trustRoot} keyword argument
            instead, since it provides the same functionality in a less
            error-prone way.

            If L{True}, do not allow anonymous sessions; defaults to L{True}.

        @param verifyOnce: If True, do not re-verify the certificate on session
            resumption.

        @param enableSingleUseKeys: If L{True}, generate a new key whenever
            ephemeral DH and ECDH parameters are used to prevent small subgroup
            attacks and to ensure perfect forward secrecy.

        @param enableSessions: This allows a shortened handshake to be used
            when a known client reconnects to the same process.  If True,
            enable OpenSSL's session caching.  Note that session caching only
            works on a single Twisted node at once.  Also, it is currently
            somewhat risky due to U{a crashing bug when using OpenSSL 1.1.1
            <https://twistedmatrix.com/trac/ticket/9764>}.

        @param fixBrokenPeers: If True, enable various non-spec protocol fixes
            for broken SSL implementations.  This should be entirely safe,
            according to the OpenSSL documentation, but YMMV.  This option is
            now off by default, because it causes problems with connections
            between peers using OpenSSL 0.9.8a.

        @param enableSessionTickets: If L{True}, enable session ticket
            extension for session resumption per RFC 5077.  Note there is no
            support for controlling session tickets.  This option is off by
            default, as some server implementations don't correctly process
            incoming empty session ticket extensions in the hello.

        @param extraCertChain: List of certificates that I{complete} your
            verification chain if the certificate authority that signed your
            C{certificate} isn't widely supported.  Do I{not} add
            C{certificate} to it.
        @type extraCertChain: C{list} of L{OpenSSL.crypto.X509}

        @param acceptableCiphers: Ciphers that are acceptable for connections.
            Uses a secure default if left L{None}.
        @type acceptableCiphers: L{IAcceptableCiphers}

        @param dhParameters: Key generation parameters that are required for
            Diffie-Hellman key exchange.  If this argument is left L{None},
            C{EDH} ciphers are I{disabled} regardless of C{acceptableCiphers}.
        @type dhParameters: L{DiffieHellmanParameters
            <twisted.internet.ssl.DiffieHellmanParameters>}

        @param trustRoot: Specification of trust requirements of peers.  If
            this argument is specified, the peer is verified.  It requires a
            certificate, and that certificate must be signed by one of the
            certificate authorities specified by this object.

            Note that since this option specifies the same information as
            C{caCerts}, C{verify}, and C{requireCertificate}, specifying any of
            those options in combination with this one will raise a
            L{TypeError}.

        @type trustRoot: L{IOpenSSLTrustRoot}

        @param acceptableProtocols: The protocols this peer is willing to speak
            after the TLS negotiation has completed, advertised over both ALPN
            and NPN.  If this argument is specified, and no overlap can be
            found with the other peer, the connection will fail to be
            established.  If the remote peer does not offer NPN or ALPN, the
            connection will be established, but no protocol wil be negotiated.
            Protocols earlier in the list are preferred over those later in the
            list.
        @type acceptableProtocols: L{list} of L{bytes}

        @param raiseMinimumTo: The minimum TLS version that you want to use, or
            Twisted's default if it is higher.  Use this if you want to make
            your client/server more secure than Twisted's default, but will
            accept Twisted's default instead if it moves higher than this
            value.  You probably want to use this over
            C{insecurelyLowerMinimumTo}.
        @type raiseMinimumTo: L{TLSVersion} constant

        @param insecurelyLowerMinimumTo: The minimum TLS version to use,
            possibly lower than Twisted's default.  If not specified, it is a
            generally considered safe default (TLSv1.0).  If you want to raise
            your minimum TLS version to above that of this default, use
            C{raiseMinimumTo}.  DO NOT use this argument unless you are
            absolutely sure this is what you want.
        @type insecurelyLowerMinimumTo: L{TLSVersion} constant

        @param lowerMaximumSecurityTo: The maximum TLS version to use.  If not
            specified, it is the most recent your OpenSSL supports.  You only
            want to set this if the peer that you are communicating with has
            problems with more recent TLS versions, it lowers your security
            when communicating with newer peers.  DO NOT use this argument
            unless you are absolutely sure this is what you want.
        @type lowerMaximumSecurityTo: L{TLSVersion} constant

        @raise ValueError: when C{privateKey} or C{certificate} are set without
            setting the respective other.
        @raise ValueError: when C{verify} is L{True} but C{caCerts} doesn't
            specify any CA certificates.
        @raise ValueError: when C{extraCertChain} is passed without specifying
            C{privateKey} or C{certificate}.
        @raise ValueError: when C{acceptableCiphers} doesn't yield any usable
            ciphers for the current platform.

        @raise TypeError: if C{trustRoot} is passed in combination with
            C{caCert}, C{verify}, or C{requireCertificate}.  Please prefer
            C{trustRoot} in new code, as its semantics are less tricky.
        @raise TypeError: if C{method} is passed in combination with
            C{tlsProtocols}.  Please prefer the more explicit C{tlsProtocols}
            in new code.

        @raises NotImplementedError: If acceptableProtocols were provided but
            no negotiation mechanism is available.
        """
        if (privateKey is None) != (certificate is None):
            raise ValueError('Specify neither or both of privateKey and certificate')
        self.privateKey = privateKey
        self.certificate = certificate
        self._options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE
        self._mode = SSL.MODE_RELEASE_BUFFERS
        if method is None:
            self.method = SSL.TLS_METHOD
            if raiseMinimumTo:
                if lowerMaximumSecurityTo and raiseMinimumTo > lowerMaximumSecurityTo:
                    raise ValueError('raiseMinimumTo needs to be lower than lowerMaximumSecurityTo')
                if raiseMinimumTo > self._defaultMinimumTLSVersion:
                    insecurelyLowerMinimumTo = raiseMinimumTo
            if insecurelyLowerMinimumTo is None:
                insecurelyLowerMinimumTo = self._defaultMinimumTLSVersion
                if lowerMaximumSecurityTo and insecurelyLowerMinimumTo > lowerMaximumSecurityTo:
                    insecurelyLowerMinimumTo = lowerMaximumSecurityTo
            if lowerMaximumSecurityTo and insecurelyLowerMinimumTo > lowerMaximumSecurityTo:
                raise ValueError('insecurelyLowerMinimumTo needs to be lower than lowerMaximumSecurityTo')
            excludedVersions = _getExcludedTLSProtocols(insecurelyLowerMinimumTo, lowerMaximumSecurityTo)
            for version in excludedVersions:
                self._options |= _tlsDisableFlags[version]
        else:
            warnings.warn('Passing method to twisted.internet.ssl.CertificateOptions was deprecated in Twisted 17.1.0. Please use a combination of insecurelyLowerMinimumTo, raiseMinimumTo, and lowerMaximumSecurityTo instead, as Twisted will correctly configure the method.', DeprecationWarning, stacklevel=3)
            self.method = method
        if verify and (not caCerts):
            raise ValueError('Specify client CA certificate information if and only if enabling certificate verification')
        self.verify = verify
        if extraCertChain is not None and None in (privateKey, certificate):
            raise ValueError('A private key and a certificate are required when adding a supplemental certificate chain.')
        if extraCertChain is not None:
            self.extraCertChain = extraCertChain
        else:
            self.extraCertChain = []
        self.caCerts = caCerts
        self.verifyDepth = verifyDepth
        self.requireCertificate = requireCertificate
        self.verifyOnce = verifyOnce
        self.enableSingleUseKeys = enableSingleUseKeys
        if enableSingleUseKeys:
            self._options |= SSL.OP_SINGLE_DH_USE | SSL.OP_SINGLE_ECDH_USE
        self.enableSessions = enableSessions
        self.fixBrokenPeers = fixBrokenPeers
        if fixBrokenPeers:
            self._options |= SSL.OP_ALL
        self.enableSessionTickets = enableSessionTickets
        if not enableSessionTickets:
            self._options |= SSL.OP_NO_TICKET
        self.dhParameters = dhParameters
        self._ecChooser = _ChooseDiffieHellmanEllipticCurve(SSL.OPENSSL_VERSION_NUMBER, openSSLlib=pyOpenSSLlib, openSSLcrypto=crypto)
        if acceptableCiphers is None:
            acceptableCiphers = defaultCiphers
        self._cipherString = ':'.join((c.fullName for c in acceptableCiphers.selectCiphers(_expandCipherString('ALL', self.method, self._options))))
        if self._cipherString == '':
            raise ValueError('Supplied IAcceptableCiphers yielded no usable ciphers on this platform.')
        if trustRoot is None:
            if self.verify:
                trustRoot = OpenSSLCertificateAuthorities(caCerts)
        else:
            self.verify = True
            self.requireCertificate = True
            trustRoot = IOpenSSLTrustRoot(trustRoot)
        self.trustRoot = trustRoot
        if acceptableProtocols is not None and (not protocolNegotiationMechanisms()):
            raise NotImplementedError('No support for protocol negotiation on this platform.')
        self._acceptableProtocols = acceptableProtocols

    def __getstate__(self):
        d = self.__dict__.copy()
        try:
            del d['_context']
        except KeyError:
            pass
        return d

    def __setstate__(self, state):
        self.__dict__ = state

    def getContext(self):
        """
        Return an L{OpenSSL.SSL.Context} object.
        """
        if self._context is None:
            self._context = self._makeContext()
        return self._context

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