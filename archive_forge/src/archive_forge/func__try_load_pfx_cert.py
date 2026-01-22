import os.path
import secrets
import ssl
import tempfile
import typing as t
def _try_load_pfx_cert(data: bytes, password: t.Optional[bytes]) -> t.Optional[t.Tuple[str, str, bytes]]:
    if not HAS_CRYPTOGRAPHY:
        return None
    try:
        pfx = pkcs12.load_key_and_certificates(data, password)
    except ValueError:
        pfx = None
    if not pfx or not pfx[0] or (not pfx[1]):
        return None
    password = password or secrets.token_bytes(32)
    certificate = pfx[1].public_bytes(encoding=serialization.Encoding.PEM).decode()
    key, password = _serialize_key_with_password(pfx[0], password)
    return (certificate, key, password)