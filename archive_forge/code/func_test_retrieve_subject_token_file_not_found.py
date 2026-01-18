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
def test_retrieve_subject_token_file_not_found(self):
    credential_source = {'file': './not_found.txt'}
    credentials = self.make_credentials(credential_source=credential_source)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.retrieve_subject_token(None)
    assert excinfo.match("File './not_found.txt' was not found")