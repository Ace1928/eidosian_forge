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
def test_retrieve_subject_token_from_url_json(self):
    credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE_JSON_URL)
    request = self.make_mock_request(token_data=JSON_FILE_CONTENT)
    subject_token = credentials.retrieve_subject_token(request)
    assert subject_token == JSON_FILE_SUBJECT_TOKEN
    self.assert_credential_request_kwargs(request.call_args_list[0][1], None)