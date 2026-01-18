import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.oauth2 import _credentials_async as _credentials_async
from google.oauth2 import credentials
from tests.oauth2 import test_credentials
def test_to_json(self):
    info = test_credentials.AUTH_USER_INFO.copy()
    creds = _credentials_async.Credentials.from_authorized_user_info(info)
    json_output = creds.to_json()
    json_asdict = json.loads(json_output)
    assert json_asdict.get('token') == creds.token
    assert json_asdict.get('refresh_token') == creds.refresh_token
    assert json_asdict.get('token_uri') == creds.token_uri
    assert json_asdict.get('client_id') == creds.client_id
    assert json_asdict.get('scopes') == creds.scopes
    assert json_asdict.get('client_secret') == creds.client_secret
    json_output = creds.to_json(strip=['client_secret'])
    json_asdict = json.loads(json_output)
    assert json_asdict.get('token') == creds.token
    assert json_asdict.get('refresh_token') == creds.refresh_token
    assert json_asdict.get('token_uri') == creds.token_uri
    assert json_asdict.get('client_id') == creds.client_id
    assert json_asdict.get('scopes') == creds.scopes
    assert json_asdict.get('client_secret') is None