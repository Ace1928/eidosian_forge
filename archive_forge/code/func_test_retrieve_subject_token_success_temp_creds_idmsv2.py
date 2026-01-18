import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
@mock.patch('google.auth._helpers.utcnow')
@mock.patch.dict(os.environ, {environment_vars.AWS_REGION: AWS_REGION, environment_vars.AWS_ACCESS_KEY_ID: ACCESS_KEY_ID, environment_vars.AWS_SECRET_ACCESS_KEY: SECRET_ACCESS_KEY})
def test_retrieve_subject_token_success_temp_creds_idmsv2(self, utcnow):
    utcnow.return_value = datetime.datetime.strptime(self.AWS_SIGNATURE_TIME, '%Y-%m-%dT%H:%M:%SZ')
    request = self.make_mock_request(role_status=http_client.OK, role_name=self.AWS_ROLE)
    credential_source_token_url = self.CREDENTIAL_SOURCE.copy()
    credential_source_token_url['imdsv2_session_token_url'] = IMDSV2_SESSION_TOKEN_URL
    credentials = self.make_credentials(credential_source=credential_source_token_url)
    credentials.retrieve_subject_token(request)
    assert not request.called