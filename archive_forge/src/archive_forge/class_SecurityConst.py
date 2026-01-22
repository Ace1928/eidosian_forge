from __future__ import absolute_import
import platform
from ctypes import (
from ctypes.util import find_library
from ...packages.six import raise_from
class SecurityConst(object):
    """
    A class object that acts as essentially a namespace for Security constants.
    """
    kSSLSessionOptionBreakOnServerAuth = 0
    kSSLProtocol2 = 1
    kSSLProtocol3 = 2
    kTLSProtocol1 = 4
    kTLSProtocol11 = 7
    kTLSProtocol12 = 8
    kTLSProtocol13 = 10
    kTLSProtocolMaxSupported = 999
    kSSLClientSide = 1
    kSSLStreamType = 0
    kSecFormatPEMSequence = 10
    kSecTrustResultInvalid = 0
    kSecTrustResultProceed = 1
    kSecTrustResultDeny = 3
    kSecTrustResultUnspecified = 4
    kSecTrustResultRecoverableTrustFailure = 5
    kSecTrustResultFatalTrustFailure = 6
    kSecTrustResultOtherError = 7
    errSSLProtocol = -9800
    errSSLWouldBlock = -9803
    errSSLClosedGraceful = -9805
    errSSLClosedNoNotify = -9816
    errSSLClosedAbort = -9806
    errSSLXCertChainInvalid = -9807
    errSSLCrypto = -9809
    errSSLInternal = -9810
    errSSLCertExpired = -9814
    errSSLCertNotYetValid = -9815
    errSSLUnknownRootCert = -9812
    errSSLNoRootCert = -9813
    errSSLHostNameMismatch = -9843
    errSSLPeerHandshakeFail = -9824
    errSSLPeerUserCancelled = -9839
    errSSLWeakPeerEphemeralDHKey = -9850
    errSSLServerAuthCompleted = -9841
    errSSLRecordOverflow = -9847
    errSecVerifyFailed = -67808
    errSecNoTrustSettings = -25263
    errSecItemNotFound = -25300
    errSecInvalidTrustSettings = -25262
    TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384 = 49196
    TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 = 49200
    TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 = 49195
    TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 = 49199
    TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256 = 52393
    TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 = 52392
    TLS_DHE_RSA_WITH_AES_256_GCM_SHA384 = 159
    TLS_DHE_RSA_WITH_AES_128_GCM_SHA256 = 158
    TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384 = 49188
    TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384 = 49192
    TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA = 49162
    TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA = 49172
    TLS_DHE_RSA_WITH_AES_256_CBC_SHA256 = 107
    TLS_DHE_RSA_WITH_AES_256_CBC_SHA = 57
    TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256 = 49187
    TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256 = 49191
    TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA = 49161
    TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA = 49171
    TLS_DHE_RSA_WITH_AES_128_CBC_SHA256 = 103
    TLS_DHE_RSA_WITH_AES_128_CBC_SHA = 51
    TLS_RSA_WITH_AES_256_GCM_SHA384 = 157
    TLS_RSA_WITH_AES_128_GCM_SHA256 = 156
    TLS_RSA_WITH_AES_256_CBC_SHA256 = 61
    TLS_RSA_WITH_AES_128_CBC_SHA256 = 60
    TLS_RSA_WITH_AES_256_CBC_SHA = 53
    TLS_RSA_WITH_AES_128_CBC_SHA = 47
    TLS_AES_128_GCM_SHA256 = 4865
    TLS_AES_256_GCM_SHA384 = 4866
    TLS_AES_128_CCM_8_SHA256 = 4869
    TLS_AES_128_CCM_SHA256 = 4868