import os.path
import secrets
import ssl
import tempfile
import typing as t
def _serialize_key_with_password(key: _PrivateKey, password: t.Optional[bytes]) -> t.Tuple[str, bytes]:
    password = password or secrets.token_bytes(32)
    return (key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.BestAvailableEncryption(password)).decode(), password)