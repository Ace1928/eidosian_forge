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
@mock.patch.dict(os.environ)
@mock.patch('google.auth._cloud_sdk.get_application_default_credentials_path', autospec=True)
def test_quota_project_from_environment(get_adc_path):
    get_adc_path.return_value = AUTHORIZED_USER_CLOUD_SDK_WITH_QUOTA_PROJECT_ID_FILE
    credentials, _ = _default.default(quota_project_id=None)
    assert credentials.quota_project_id == 'quota_project_id'
    quota_from_env = 'quota_from_env'
    os.environ[environment_vars.GOOGLE_CLOUD_QUOTA_PROJECT] = quota_from_env
    credentials, _ = _default.default(quota_project_id=None)
    assert credentials.quota_project_id == quota_from_env
    explicit_quota = 'explicit_quota'
    credentials, _ = _default.default(quota_project_id=explicit_quota)
    assert credentials.quota_project_id == explicit_quota