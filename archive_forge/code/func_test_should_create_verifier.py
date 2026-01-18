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
def test_should_create_verifier(self):
    image_props = {CERT_UUID: 'CERT_UUID', HASH_METHOD: 'HASH_METHOD', SIGNATURE: 'SIGNATURE', KEY_TYPE: 'SIG_KEY_TYPE'}
    self.assertTrue(signature_utils.should_create_verifier(image_props))