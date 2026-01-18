import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
from google.auth import transport
def test_with_invalid_impersonation_target_principal(self):
    invalid_url = 'https://iamcredentials.googleapis.com/v1/invalid'
    with pytest.raises(exceptions.RefreshError) as excinfo:
        self.make_credentials(service_account_impersonation_url=invalid_url)
    assert excinfo.match('Unable to determine target principal from service account impersonation URL.')