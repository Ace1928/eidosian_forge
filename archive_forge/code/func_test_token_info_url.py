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
def test_token_info_url(self):
    credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE.copy())
    assert credentials.token_info_url == TOKEN_INFO_URL