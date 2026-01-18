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
def test_pickle_with_missing_attribute(self):
    creds = self.make_credentials()
    del creds.__dict__['_quota_project_id']
    unpickled = pickle.loads(pickle.dumps(creds))
    assert unpickled.quota_project_id is None