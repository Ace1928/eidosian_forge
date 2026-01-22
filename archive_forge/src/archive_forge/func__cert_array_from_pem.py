import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security
def _cert_array_from_pem(pem_bundle):
    """
    Given a bundle of certs in PEM format, turns them into a CFArray of certs
    that can be used to validate a cert chain.
    """
    pem_bundle = pem_bundle.replace(b'\r\n', b'\n')
    der_certs = [base64.b64decode(match.group(1)) for match in _PEM_CERTS_RE.finditer(pem_bundle)]
    if not der_certs:
        raise ssl.SSLError('No root certificates specified')
    cert_array = CoreFoundation.CFArrayCreateMutable(CoreFoundation.kCFAllocatorDefault, 0, ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks))
    if not cert_array:
        raise ssl.SSLError('Unable to allocate memory!')
    try:
        for der_bytes in der_certs:
            certdata = _cf_data_from_bytes(der_bytes)
            if not certdata:
                raise ssl.SSLError('Unable to allocate memory!')
            cert = Security.SecCertificateCreateWithData(CoreFoundation.kCFAllocatorDefault, certdata)
            CoreFoundation.CFRelease(certdata)
            if not cert:
                raise ssl.SSLError('Unable to build cert object!')
            CoreFoundation.CFArrayAppendValue(cert_array, cert)
            CoreFoundation.CFRelease(cert)
    except Exception:
        CoreFoundation.CFRelease(cert_array)
        raise
    return cert_array