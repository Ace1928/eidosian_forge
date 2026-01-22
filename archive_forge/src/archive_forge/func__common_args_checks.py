from __future__ import annotations
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized, InvalidKey
from cryptography.hazmat.primitives import constant_time, hashes, hmac
from cryptography.hazmat.primitives.kdf import KeyDerivationFunction
def _common_args_checks(algorithm: hashes.HashAlgorithm, length: int, otherinfo: typing.Optional[bytes]) -> None:
    max_length = algorithm.digest_size * (2 ** 32 - 1)
    if length > max_length:
        raise ValueError(f'Cannot derive keys larger than {max_length} bits.')
    if otherinfo is not None:
        utils._check_bytes('otherinfo', otherinfo)