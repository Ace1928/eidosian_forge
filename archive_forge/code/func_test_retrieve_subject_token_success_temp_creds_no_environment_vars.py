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
def test_retrieve_subject_token_success_temp_creds_no_environment_vars(self, utcnow):
    utcnow.return_value = datetime.datetime.strptime(self.AWS_SIGNATURE_TIME, '%Y-%m-%dT%H:%M:%SZ')
    request = self.make_mock_request(region_status=http_client.OK, region_name=self.AWS_REGION, role_status=http_client.OK, role_name=self.AWS_ROLE, security_credentials_status=http_client.OK, security_credentials_data=self.AWS_SECURITY_CREDENTIALS_RESPONSE)
    credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE)
    subject_token = credentials.retrieve_subject_token(request)
    assert subject_token == self.make_serialized_aws_signed_request({'access_key_id': ACCESS_KEY_ID, 'secret_access_key': SECRET_ACCESS_KEY, 'security_token': TOKEN})
    self.assert_aws_metadata_request_kwargs(request.call_args_list[0][1], REGION_URL)
    self.assert_aws_metadata_request_kwargs(request.call_args_list[1][1], SECURITY_CREDS_URL)
    self.assert_aws_metadata_request_kwargs(request.call_args_list[2][1], '{}/{}'.format(SECURITY_CREDS_URL, self.AWS_ROLE), {'Content-Type': 'application/json'})
    new_request = self.make_mock_request(role_status=http_client.OK, role_name=self.AWS_ROLE, security_credentials_status=http_client.OK, security_credentials_data=self.AWS_SECURITY_CREDENTIALS_RESPONSE)
    credentials.retrieve_subject_token(new_request)
    assert len(new_request.call_args_list) == 2
    self.assert_aws_metadata_request_kwargs(new_request.call_args_list[0][1], SECURITY_CREDS_URL)
    self.assert_aws_metadata_request_kwargs(new_request.call_args_list[1][1], '{}/{}'.format(SECURITY_CREDS_URL, self.AWS_ROLE), {'Content-Type': 'application/json'})