import warnings
import base64
import typing as t
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import UnsupportedAlgorithm
from requests.auth import AuthBase
from requests.packages.urllib3.response import HTTPResponse
import spnego
class ShimSessionSecurity:
    """Shim used for backwards compatibility with ntlm-auth."""

    def __init__(self, context: spnego.ContextProxy) -> None:
        self._context = context

    def wrap(self, message) -> t.Tuple[bytes, bytes]:
        wrap_res = self._context.wrap(message, encrypt=True)
        signature = wrap_res.data[:16]
        data = wrap_res.data[16:]
        return (data, signature)

    def unwrap(self, message: bytes, signature: bytes) -> bytes:
        data = signature + message
        return self._context.unwrap(data).data

    def get_signature(self, message: bytes) -> bytes:
        return self._context.sign(message)

    def verify_signature(self, message: bytes, signature: bytes) -> None:
        self._context.verify(message, signature)