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
class FakeCryptoCertificate(object):

    def __init__(self, pub_key=TEST_RSA_PRIVATE_KEY.public_key(), not_valid_before=timeutils.utcnow() - datetime.timedelta(hours=1), not_valid_after=timeutils.utcnow() + datetime.timedelta(hours=2)):
        self.pub_key = pub_key
        self.cert_not_valid_before = not_valid_before
        self.cert_not_valid_after = not_valid_after

    def public_key(self):
        return self.pub_key

    @property
    def not_valid_before(self):
        return self.cert_not_valid_before

    @property
    def not_valid_after(self):
        return self.cert_not_valid_after