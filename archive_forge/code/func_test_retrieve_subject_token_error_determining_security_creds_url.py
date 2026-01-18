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
def test_retrieve_subject_token_error_determining_security_creds_url(self):
    credential_source = self.CREDENTIAL_SOURCE.copy()
    credential_source.pop('url')
    request = self.make_mock_request(region_status=http_client.OK, region_name=self.AWS_REGION)
    credentials = self.make_credentials(credential_source=credential_source)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.retrieve_subject_token(request)
    assert excinfo.match('Unable to determine the AWS metadata server security credentials endpoint')