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
def test_refresh_no_refresh_token(self):
    request = mock.create_autospec(transport.Request)
    credentials_ = credentials.Credentials(token=None, refresh_token=None)
    with pytest.raises(exceptions.RefreshError, match='necessary fields'):
        credentials_.refresh(request)
    request.assert_not_called()