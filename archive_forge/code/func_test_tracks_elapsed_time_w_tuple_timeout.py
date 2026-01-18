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
def test_tracks_elapsed_time_w_tuple_timeout(self, frozen_time):
    with self.make_guard(timeout=(16, 19)) as guard:
        frozen_time.tick(delta=datetime.timedelta(seconds=3.8))
    assert guard.remaining_timeout == (12.2, 15.2)