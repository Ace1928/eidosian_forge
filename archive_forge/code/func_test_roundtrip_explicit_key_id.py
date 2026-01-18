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
def test_roundtrip_explicit_key_id(token_factory):
    token = token_factory(key_id='3')
    certs = {'2': OTHER_CERT_BYTES, '3': PUBLIC_CERT_BYTES}
    payload = jwt.decode(token, certs)
    assert payload['user'] == 'billy bob'