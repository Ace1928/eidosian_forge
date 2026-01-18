import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
def test_token_url_custom(self):
    for url in VALID_TOKEN_URLS:
        credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE.copy(), token_url=url + '/token')
        assert credentials._token_url == url + '/token'