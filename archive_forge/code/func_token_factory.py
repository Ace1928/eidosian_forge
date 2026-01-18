import base64
import datetime
import json
import os
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import jwt
@pytest.fixture
def token_factory(signer, es256_signer):

    def factory(claims=None, key_id=None, use_es256_signer=False):
        now = _helpers.datetime_to_secs(_helpers.utcnow())
        payload = {'aud': 'audience@example.com', 'iat': now, 'exp': now + 300, 'user': 'billy bob', 'metadata': {'meta': 'data'}}
        payload.update(claims or {})
        if key_id is False:
            signer._key_id = None
            key_id = None
        if use_es256_signer:
            return jwt.encode(es256_signer, payload, key_id=key_id)
        else:
            return jwt.encode(signer, payload, key_id=key_id)
    return factory