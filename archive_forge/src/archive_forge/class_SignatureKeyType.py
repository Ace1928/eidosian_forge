import binascii
from castellan.common.exception import KeyManagerError
from castellan.common.exception import ManagedObjectNotFoundError
from castellan import key_manager
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography import x509
from oslo_log import log as logging
from oslo_serialization import base64
from oslo_utils import encodeutils
from cursive import exception
from cursive.i18n import _, _LE
from cursive import verifiers
class SignatureKeyType(object):
    REGISTERED_TYPES = {}

    def __init__(self, name, public_key_type, create_verifier):
        self.name = name
        self.public_key_type = public_key_type
        self.create_verifier = create_verifier

    @classmethod
    def register(cls, name, public_key_type, create_verifier):
        """Register a signature key type.

        :param name: the name of the signature key type
        :param public_key_type: e.g. RSAPublicKey, DSAPublicKey, etc.
        :param create_verifier: a function to create a verifier for this type
        """
        cls.REGISTERED_TYPES[name] = cls(name, public_key_type, create_verifier)

    @classmethod
    def lookup(cls, name):
        """Look up the signature key type.

        :param name: the name of the signature key type
        :returns: the SignatureKeyType object
        :raises: SignatureVerificationError if signature key type is invalid
        """
        if name not in cls.REGISTERED_TYPES:
            raise exception.SignatureVerificationError(reason=_('Invalid signature key type: %s') % name)
        return cls.REGISTERED_TYPES[name]