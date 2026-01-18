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
def test_refresh_failure_unauthorzed(self, mock_donor_credentials):
    credentials = self.make_credentials(lifetime=None)
    response_body = {'error': {'code': 403, 'message': 'The caller does not have permission', 'status': 'PERMISSION_DENIED'}}
    request = self.make_request(data=json.dumps(response_body), status=http_client.UNAUTHORIZED)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.refresh(request)
    assert excinfo.match(impersonated_credentials._REFRESH_ERROR)
    assert not credentials.valid
    assert credentials.expired