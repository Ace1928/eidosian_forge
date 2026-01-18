from __future__ import print_function
import base64
import hashlib
import os
from cStringIO import StringIO
from M2Crypto import BIO, EVP, RSA, X509, m2
def x509_verify(cacert, cert, binary=False):
    """Validate the certificate's authenticity using a certification authority"""
    ca = x509_parse_cert(cacert)
    crt = x509_parse_cert(cert, binary)
    return crt.verify(ca.get_pubkey())