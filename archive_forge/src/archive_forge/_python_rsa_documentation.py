from __future__ import absolute_import
from pyasn1.codec.der import decoder  # type: ignore
from pyasn1_modules import pem  # type: ignore
from pyasn1_modules.rfc2459 import Certificate  # type: ignore
from pyasn1_modules.rfc5208 import PrivateKeyInfo  # type: ignore
import rsa  # type: ignore
import six
from google.auth import _helpers
from google.auth import exceptions
from google.auth.crypt import base
Construct an Signer instance from a private key in PEM format.

        Args:
            key (str): Private key in PEM format.
            key_id (str): An optional key id used to identify the private key.

        Returns:
            google.auth.crypt.Signer: The constructed signer.

        Raises:
            ValueError: If the key cannot be parsed as PKCS#1 or PKCS#8 in
                PEM format.
        