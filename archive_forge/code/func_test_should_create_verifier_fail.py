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
def test_should_create_verifier_fail(self):
    bad_image_properties = [{CERT_UUID: 'CERT_UUID', HASH_METHOD: 'HASH_METHOD', SIGNATURE: 'SIGNATURE'}, {CERT_UUID: 'CERT_UUID', HASH_METHOD: 'HASH_METHOD', KEY_TYPE: 'SIG_KEY_TYPE'}, {CERT_UUID: 'CERT_UUID', SIGNATURE: 'SIGNATURE', KEY_TYPE: 'SIG_KEY_TYPE'}, {HASH_METHOD: 'HASH_METHOD', SIGNATURE: 'SIGNATURE', KEY_TYPE: 'SIG_KEY_TYPE'}]
    for bad_props in bad_image_properties:
        result = signature_utils.should_create_verifier(bad_props)
        self.assertFalse(result)