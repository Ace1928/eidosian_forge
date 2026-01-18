import json
import os
import mock
import pytest  # type: ignore
from google.auth import _default
from google.auth import api_key
from google.auth import app_engine
from google.auth import aws
from google.auth import compute_engine
from google.auth import credentials
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import impersonated_credentials
from google.auth import pluggable
from google.oauth2 import gdch_credentials
from google.oauth2 import service_account
import google.oauth2.credentials
@mock.patch('google.auth.compute_engine._metadata.ping', return_value=True, autospec=True)
@mock.patch('google.auth.compute_engine._metadata.get_project_id', return_value='example-project', autospec=True)
@mock.patch.dict(os.environ)
def test_quota_gce_credentials(unused_get, unused_ping):
    credentials, project_id = _default._get_gce_credentials()
    assert project_id == 'example-project'
    assert credentials.quota_project_id is None
    quota_from_env = 'quota_from_env'
    os.environ[environment_vars.GOOGLE_CLOUD_QUOTA_PROJECT] = quota_from_env
    credentials, project_id = _default._get_gce_credentials()
    assert credentials.quota_project_id == quota_from_env
    explicit_quota = 'explicit_quota'
    credentials, project_id = _default._get_gce_credentials(quota_project_id=explicit_quota)
    assert credentials.quota_project_id == explicit_quota