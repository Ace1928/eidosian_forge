from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
def _get_rsa_pss_salt_length(backend: Backend, pss: PSS, key: typing.Union[RSAPrivateKey, RSAPublicKey], hash_algorithm: hashes.HashAlgorithm) -> int:
    salt = pss._salt_length
    if isinstance(salt, _MaxLength):
        return calculate_max_pss_salt_length(key, hash_algorithm)
    elif isinstance(salt, _DigestLength):
        return hash_algorithm.digest_size
    elif isinstance(salt, _Auto):
        if isinstance(key, RSAPrivateKey):
            raise ValueError('PSS salt length can only be set to AUTO when verifying')
        return backend._lib.RSA_PSS_SALTLEN_AUTO
    else:
        return salt