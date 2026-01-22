import cryptography.exceptions
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
import cryptography.x509
from google.auth import _helpers
from google.auth.crypt import base
class RSASigner(base.Signer, base.FromServiceAccountMixin):
    """Signs messages with an RSA private key.

    Args:
        private_key (
                cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey):
            The private key to sign with.
        key_id (str): Optional key ID used to identify this private key. This
            can be useful to associate the private key with its associated
            public key or certificate.
    """

    def __init__(self, private_key, key_id=None):
        self._key = private_key
        self._key_id = key_id

    @property
    @_helpers.copy_docstring(base.Signer)
    def key_id(self):
        return self._key_id

    @_helpers.copy_docstring(base.Signer)
    def sign(self, message):
        message = _helpers.to_bytes(message)
        return self._key.sign(message, _PADDING, _SHA256)

    @classmethod
    def from_string(cls, key, key_id=None):
        """Construct a RSASigner from a private key in PEM format.

        Args:
            key (Union[bytes, str]): Private key in PEM format.
            key_id (str): An optional key id used to identify the private key.

        Returns:
            google.auth.crypt._cryptography_rsa.RSASigner: The
            constructed signer.

        Raises:
            ValueError: If ``key`` is not ``bytes`` or ``str`` (unicode).
            UnicodeDecodeError: If ``key`` is ``bytes`` but cannot be decoded
                into a UTF-8 ``str``.
            ValueError: If ``cryptography`` "Could not deserialize key data."
        """
        key = _helpers.to_bytes(key)
        private_key = serialization.load_pem_private_key(key, password=None, backend=_BACKEND)
        return cls(private_key, key_id=key_id)