import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import exceptions
from google.auth import external_account_authorized_user
from google.auth import transport
def test_stunted_create_no_client_id(self):
    with pytest.raises(ValueError) as excinfo:
        self.make_credentials(token=None, client_id=None)
    assert excinfo.match('Token should be created with fields to make it valid \\(`token` and `expiry`\\), or fields to allow it to refresh \\(`refresh_token`, `token_url`, `client_id`, `client_secret`\\)\\.')