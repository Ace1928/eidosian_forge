from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Util.asn1 import DerSequence
from oauth2client_4_0 import _helpers
class PyCryptoSigner(object):
    """Signs messages with a private key."""

    def __init__(self, pkey):
        """Constructor.

        Args:
            pkey, OpenSSL.crypto.PKey (or equiv), The private key to sign with.
        """
        self._key = pkey

    def sign(self, message):
        """Signs a message.

        Args:
            message: string, Message to be signed.

        Returns:
            string, The signature of the message for the given key.
        """
        message = _helpers._to_bytes(message, encoding='utf-8')
        return PKCS1_v1_5.new(self._key).sign(SHA256.new(message))

    @staticmethod
    def from_string(key, password='notasecret'):
        """Construct a Signer instance from a string.

        Args:
            key: string, private key in PEM format.
            password: string, password for private key file. Unused for PEM
                      files.

        Returns:
            Signer instance.

        Raises:
            NotImplementedError if the key isn't in PEM format.
        """
        parsed_pem_key = _helpers._parse_pem_key(_helpers._to_bytes(key))
        if parsed_pem_key:
            pkey = RSA.importKey(parsed_pem_key)
        else:
            raise NotImplementedError('No key in PEM format was detected. This implementation can only use the PyCrypto library for keys in PEM format.')
        return PyCryptoSigner(pkey)