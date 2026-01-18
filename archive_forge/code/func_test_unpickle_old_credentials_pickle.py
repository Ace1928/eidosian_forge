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
@pytest.mark.skipif(sys.version_info < (3, 5), reason='pickle file can only be loaded with Python >= 3.5')
def test_unpickle_old_credentials_pickle(self):
    with open(os.path.join(test_credentials.DATA_DIR, 'old_oauth_credentials_py3.pickle'), 'rb') as f:
        credentials = pickle.load(f)
        assert credentials.quota_project_id is None