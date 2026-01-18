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
@pytest.mark.parametrize('time_skew', [100, -100])
def test_refresh_source_credentials(self, time_skew):
    credentials = self.make_credentials(lifetime=None)
    credentials._source_credentials.expiry = _helpers.utcnow() + _helpers.REFRESH_THRESHOLD + datetime.timedelta(seconds=time_skew)
    credentials._source_credentials.token = 'Token'
    with mock.patch('google.oauth2.service_account.Credentials.refresh', autospec=True) as source_cred_refresh:
        expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
        response_body = {'accessToken': 'token', 'expireTime': expire_time}
        request = self.make_request(data=json.dumps(response_body), status=http_client.OK)
        credentials.refresh(request)
        assert credentials.valid
        assert not credentials.expired
        if time_skew > 0:
            source_cred_refresh.assert_not_called()
        else:
            source_cred_refresh.assert_called_once()