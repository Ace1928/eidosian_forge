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
def test_retrieve_subject_token_invalid_json(self, tmpdir):
    invalid_json_file = tmpdir.join('invalid.json')
    invalid_json_file.write('{')
    credential_source = {'file': str(invalid_json_file), 'format': {'type': 'json', 'subject_token_field_name': 'access_token'}}
    credentials = self.make_credentials(credential_source=credential_source)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.retrieve_subject_token(None)
    assert excinfo.match("Unable to parse subject_token from JSON file '{}' using key '{}'".format(str(invalid_json_file), 'access_token'))