from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
def _enc_dec_rsa(backend: Backend, key: typing.Union[_RSAPrivateKey, _RSAPublicKey], data: bytes, padding: AsymmetricPadding) -> bytes:
    if not isinstance(padding, AsymmetricPadding):
        raise TypeError('Padding must be an instance of AsymmetricPadding.')
    if isinstance(padding, PKCS1v15):
        padding_enum = backend._lib.RSA_PKCS1_PADDING
    elif isinstance(padding, OAEP):
        padding_enum = backend._lib.RSA_PKCS1_OAEP_PADDING
        if not isinstance(padding._mgf, MGF1):
            raise UnsupportedAlgorithm('Only MGF1 is supported by this backend.', _Reasons.UNSUPPORTED_MGF)
        if not backend.rsa_padding_supported(padding):
            raise UnsupportedAlgorithm('This combination of padding and hash algorithm is not supported by this backend.', _Reasons.UNSUPPORTED_PADDING)
    else:
        raise UnsupportedAlgorithm(f'{padding.name} is not supported by this backend.', _Reasons.UNSUPPORTED_PADDING)
    return _enc_dec_rsa_pkey_ctx(backend, key, data, padding_enum, padding)