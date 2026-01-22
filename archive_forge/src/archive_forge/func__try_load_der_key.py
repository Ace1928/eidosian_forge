import os.path
import secrets
import ssl
import tempfile
import typing as t
def _try_load_der_key(data: bytes, password: t.Optional[bytes]) -> t.Optional[t.Tuple[str, bytes]]:
    if not HAS_CRYPTOGRAPHY:
        return None
    try:
        key = serialization.load_der_private_key(data, password=password)
    except ValueError:
        return None
    else:
        return _serialize_key_with_password(key, password)