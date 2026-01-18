import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import sts
from google.oauth2 import utils
def test_refresh_token_success_with_refresh(self):
    """Test refresh token with successful response."""
    client = self.make_client(self.CLIENT_AUTH_BASIC)
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE_WITH_REFRESH)
    response = client.refresh_token(request, 'refreshtoken')
    headers = {'Authorization': 'Basic dXNlcm5hbWU6cGFzc3dvcmQ=', 'Content-Type': 'application/x-www-form-urlencoded'}
    request_data = {'grant_type': 'refresh_token', 'refresh_token': 'refreshtoken'}
    self.assert_request_kwargs(request.call_args[1], headers, request_data)
    assert response == self.SUCCESS_RESPONSE_WITH_REFRESH