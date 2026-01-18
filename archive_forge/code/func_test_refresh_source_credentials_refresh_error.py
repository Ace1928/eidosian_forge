import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import credentials
from google.auth import downscoped
from google.auth import exceptions
from google.auth import transport
def test_refresh_source_credentials_refresh_error(self):
    credentials = self.make_credentials(source_credentials=SourceCredentials(raise_error=True))
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.refresh(mock.sentinel.request)
    assert excinfo.match('Failed to refresh access token in source credentials.')
    assert not credentials.expired
    assert credentials.token is None