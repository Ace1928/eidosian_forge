from OpenSSL.crypto import PKey, X509
from cryptography import x509
from cryptography.hazmat.primitives.serialization import (load_pem_private_key,
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.backends import default_backend
from datetime import datetime
from requests.adapters import HTTPAdapter
import requests
from .. import exceptions as exc
importing the protocol constants from _ssl instead of ssl because only the
def _import_pyopensslcontext(self):
    global PyOpenSSLContext
    if requests.__build__ < 135680:
        PyOpenSSLContext = None
    else:
        try:
            from requests.packages.urllib3.contrib.pyopenssl import PyOpenSSLContext
        except ImportError:
            try:
                from urllib3.contrib.pyopenssl import PyOpenSSLContext
            except ImportError:
                PyOpenSSLContext = None