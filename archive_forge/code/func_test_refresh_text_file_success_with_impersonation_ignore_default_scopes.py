import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import identity_pool
from google.auth import transport
def test_refresh_text_file_success_with_impersonation_ignore_default_scopes(self):
    credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE_TEXT, service_account_impersonation_url=SERVICE_ACCOUNT_IMPERSONATION_URL, scopes=SCOPES, default_scopes=['ignored'])
    self.assert_underlying_credentials_refresh(credentials=credentials, audience=AUDIENCE, subject_token=TEXT_FILE_SUBJECT_TOKEN, subject_token_type=SUBJECT_TOKEN_TYPE, token_url=TOKEN_URL, service_account_impersonation_url=SERVICE_ACCOUNT_IMPERSONATION_URL, basic_auth_encoding=None, quota_project_id=None, used_scopes=SCOPES, scopes=SCOPES, default_scopes=['ignored'])