import base64
import json
import os
from cryptography.hazmat.primitives.asymmetric import ec
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import base
from google.auth.crypt import es256
def test_verify_failure_with_wrong_raw_signature(self):
    to_sign = b'foo'
    wrong_signature = base64.urlsafe_b64decode(b'm7oaRxUDeYqjZ8qiMwo0PZLTMZWKJLFQREpqce1StMIa_yXQQ-C5WgeIRHW7OqlYSDL0XbUrj_uAw9i-QhfOJQ==')
    verifier = es256.ES256Verifier.from_string(PUBLIC_KEY_BYTES)
    assert not verifier.verify(to_sign, wrong_signature)