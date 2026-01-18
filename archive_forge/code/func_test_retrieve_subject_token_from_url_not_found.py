import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import identity_pool
from google.auth import transport
def test_retrieve_subject_token_from_url_not_found(self):
    credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE_TEXT_URL)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.retrieve_subject_token(self.make_mock_request(token_status=404, token_data=JSON_FILE_CONTENT))
    assert excinfo.match('Unable to retrieve Identity Pool subject token')