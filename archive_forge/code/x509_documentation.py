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
Create an SSL Context with the supplied cert/password.

    :param cert_bytes array of bytes containing the cert encoded
           using the method supplied in the ``encoding`` parameter
    :param pk_bytes array of bytes containing the private key encoded
           using the method supplied in the ``encoding`` parameter
    :param password array of bytes containing the passphrase to be used
           with the supplied private key. None if unencrypted.
           Defaults to None.
    :param encoding ``cryptography.hazmat.primitives.serialization.Encoding``
            details the encoding method used on the ``cert_bytes``  and
            ``pk_bytes`` parameters. Can be either PEM or DER.
            Defaults to PEM.
    