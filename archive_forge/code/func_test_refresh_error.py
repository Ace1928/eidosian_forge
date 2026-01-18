import base64
import datetime
import mock
import pytest  # type: ignore
import responses  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.auth.compute_engine import credentials
from google.auth.transport import requests
@mock.patch('google.auth._helpers.utcnow', return_value=datetime.datetime.utcfromtimestamp(0))
@mock.patch('google.auth.compute_engine._metadata.get', autospec=True)
@mock.patch('google.auth.iam.Signer.sign', autospec=True)
def test_refresh_error(self, sign, get, utcnow):
    get.side_effect = [{'email': 'service-account@example.com', 'scopes': ['one', 'two']}]
    sign.side_effect = [b'signature']
    request = mock.create_autospec(transport.Request, instance=True)
    response = mock.Mock()
    response.data = b'{"error": "http error"}'
    response.status = 404
    request.side_effect = [response]
    self.credentials = credentials.IDTokenCredentials(request=request, target_audience='https://audience.com')
    with pytest.raises(exceptions.RefreshError) as excinfo:
        self.credentials.refresh(request)
    assert excinfo.match('http error')