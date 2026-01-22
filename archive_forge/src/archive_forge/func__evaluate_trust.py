from __future__ import absolute_import
import contextlib
import ctypes
import errno
import os.path
import shutil
import socket
import ssl
import struct
import threading
import weakref
import six
from .. import util
from ..util.ssl_ import PROTOCOL_TLS_CLIENT
from ._securetransport.bindings import CoreFoundation, Security, SecurityConst
from ._securetransport.low_level import (
def _evaluate_trust(self, trust_bundle):
    if os.path.isfile(trust_bundle):
        with open(trust_bundle, 'rb') as f:
            trust_bundle = f.read()
    cert_array = None
    trust = Security.SecTrustRef()
    try:
        cert_array = _cert_array_from_pem(trust_bundle)
        result = Security.SSLCopyPeerTrust(self.context, ctypes.byref(trust))
        _assert_no_error(result)
        if not trust:
            raise ssl.SSLError('Failed to copy trust reference')
        result = Security.SecTrustSetAnchorCertificates(trust, cert_array)
        _assert_no_error(result)
        result = Security.SecTrustSetAnchorCertificatesOnly(trust, True)
        _assert_no_error(result)
        trust_result = Security.SecTrustResultType()
        result = Security.SecTrustEvaluate(trust, ctypes.byref(trust_result))
        _assert_no_error(result)
    finally:
        if trust:
            CoreFoundation.CFRelease(trust)
        if cert_array is not None:
            CoreFoundation.CFRelease(cert_array)
    return trust_result.value