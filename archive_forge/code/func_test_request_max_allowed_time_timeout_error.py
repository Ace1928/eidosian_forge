import datetime
import functools
import os
import sys
import freezegun
import mock
import OpenSSL
import pytest  # type: ignore
import requests
import requests.adapters
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._custom_tls_signer
import google.auth.transport._mtls_helper
import google.auth.transport.requests
from google.oauth2 import service_account
from tests.transport import compliance
def test_request_max_allowed_time_timeout_error(self, frozen_time):
    tick_one_second = functools.partial(frozen_time.tick, delta=datetime.timedelta(seconds=1.0))
    credentials = mock.Mock(wraps=TimeTickCredentialsStub(time_tick=tick_one_second))
    adapter = TimeTickAdapterStub(time_tick=tick_one_second, responses=[make_response(status=http_client.OK)])
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials)
    authed_session.mount(self.TEST_URL, adapter)
    with pytest.raises(requests.exceptions.Timeout):
        authed_session.request('GET', self.TEST_URL, max_allowed_time=0.9)