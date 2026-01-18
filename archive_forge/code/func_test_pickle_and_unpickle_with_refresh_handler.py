import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import credentials
def test_pickle_and_unpickle_with_refresh_handler(self):
    expected_expiry = _helpers.utcnow() + datetime.timedelta(seconds=2800)
    refresh_handler = mock.Mock(return_value=('TOKEN', expected_expiry))
    creds = credentials.Credentials(token=None, refresh_token=None, token_uri=None, client_id=None, client_secret=None, rapt_token=None, refresh_handler=refresh_handler)
    unpickled = pickle.loads(pickle.dumps(creds))
    assert list(creds.__dict__).sort() == list(unpickled.__dict__).sort()
    for attr in list(creds.__dict__):
        if attr == '_refresh_handler':
            assert getattr(unpickled, attr) is None
        else:
            assert getattr(creds, attr) == getattr(unpickled, attr)