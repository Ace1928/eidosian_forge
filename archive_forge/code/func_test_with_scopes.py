import base64
import datetime
import mock
import pytest  # type: ignore
import responses  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.auth.compute_engine import credentials
from google.auth.transport import requests
def test_with_scopes(self):
    assert self.credentials._scopes is None
    scopes = ['one', 'two']
    self.credentials = self.credentials.with_scopes(scopes)
    assert self.credentials._scopes == scopes