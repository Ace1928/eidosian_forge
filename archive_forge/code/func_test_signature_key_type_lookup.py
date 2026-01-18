import base64
import datetime
import mock
from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
import cryptography.exceptions as crypto_exceptions
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from oslo_utils import timeutils
from cursive import exception
from cursive import signature_utils
from cursive.tests import base
def test_signature_key_type_lookup(self):
    for sig_format in [signature_utils.RSA_PSS, signature_utils.DSA]:
        sig_key_type = signature_utils.SignatureKeyType.lookup(sig_format)
        self.assertIsInstance(sig_key_type, signature_utils.SignatureKeyType)
        self.assertEqual(sig_format, sig_key_type.name)