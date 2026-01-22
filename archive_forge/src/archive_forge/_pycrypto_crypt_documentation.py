from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Util.asn1 import DerSequence
from oauth2client_4_0 import _helpers
Construct a Signer instance from a string.

        Args:
            key: string, private key in PEM format.
            password: string, password for private key file. Unused for PEM
                      files.

        Returns:
            Signer instance.

        Raises:
            NotImplementedError if the key isn't in PEM format.
        