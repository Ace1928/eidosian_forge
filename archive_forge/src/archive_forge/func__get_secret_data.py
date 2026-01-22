import calendar
import time
import urllib
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization
from cryptography import x509 as cryptography_x509
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import service_token
from keystoneauth1 import session
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from castellan.common import exception
from castellan.common.objects import key as key_base_class
from castellan.common.objects import opaque_data as op_data
from castellan.i18n import _
from castellan.key_manager import key_manager
from barbicanclient import client as barbican_client_import
from barbicanclient import exceptions as barbican_exceptions
from oslo_utils import timeutils
def _get_secret_data(self, secret):
    """Retrieves the secret data.

        Converts the Barbican secret to bytes suitable for a Castellan object.
        If the secret is a public key, private key, or certificate, the secret
        is expected to be in PEM format and will be converted to DER.

        :param secret: the secret from barbican with the payload of data
        :returns: the secret data
        """
    if secret.secret_type == 'public':
        key = serialization.load_pem_public_key(secret.payload, backend=backends.default_backend())
        return key.public_bytes(encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo)
    elif secret.secret_type == 'private':
        key = serialization.load_pem_private_key(secret.payload, backend=backends.default_backend(), password=None)
        return key.private_bytes(encoding=serialization.Encoding.DER, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
    elif secret.secret_type == 'certificate':
        cert = cryptography_x509.load_pem_x509_certificate(secret.payload, backend=backends.default_backend())
        return cert.public_bytes(encoding=serialization.Encoding.DER)
    else:
        return secret.payload