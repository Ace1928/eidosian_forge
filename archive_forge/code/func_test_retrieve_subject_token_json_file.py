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
def test_retrieve_subject_token_json_file(self):
    credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE_JSON)
    subject_token = credentials.retrieve_subject_token(None)
    assert subject_token == JSON_FILE_SUBJECT_TOKEN