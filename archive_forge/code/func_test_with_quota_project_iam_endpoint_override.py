import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import transport
from google.auth.impersonated_credentials import Credentials
from google.oauth2 import credentials
from google.oauth2 import service_account
@pytest.mark.parametrize('use_data_bytes', [True, False])
def test_with_quota_project_iam_endpoint_override(self, use_data_bytes, mock_donor_credentials):
    credentials = self.make_credentials(lifetime=None, iam_endpoint_override=self.IAM_ENDPOINT_OVERRIDE)
    token = 'token'
    quota_project_creds = credentials.with_quota_project('project-foo')
    expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
    response_body = {'accessToken': token, 'expireTime': expire_time}
    request = self.make_request(data=json.dumps(response_body), status=http_client.OK, use_data_bytes=use_data_bytes)
    quota_project_creds.refresh(request)
    assert quota_project_creds.valid
    assert not quota_project_creds.expired
    request_kwargs = request.call_args[1]
    assert request_kwargs['url'] == self.IAM_ENDPOINT_OVERRIDE