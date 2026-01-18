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
def test_nonworkforce_with_workforce_pool_user_project(self):
    with pytest.raises(ValueError) as excinfo:
        CredentialsImpl(audience=self.AUDIENCE, subject_token_type=self.SUBJECT_TOKEN_TYPE, token_url=self.TOKEN_URL, credential_source=self.CREDENTIAL_SOURCE, workforce_pool_user_project=self.WORKFORCE_POOL_USER_PROJECT)
    assert excinfo.match('workforce_pool_user_project should not be set for non-workforce pool credentials')