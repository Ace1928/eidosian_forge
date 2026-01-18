import base64
import json
import os
from cryptography.hazmat.primitives.asymmetric import ec
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import base
from google.auth.crypt import es256
def test_verify_failure(self):
    verifier = es256.ES256Verifier.from_string(PUBLIC_KEY_BYTES)
    bad_signature1 = b''
    assert not verifier.verify(b'foo', bad_signature1)
    bad_signature2 = b'a'
    assert not verifier.verify(b'foo', bad_signature2)