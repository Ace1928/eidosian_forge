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
def test_noop_if_no_timeout(self, frozen_time):
    with self.make_guard(timeout=None) as guard:
        frozen_time.tick(delta=datetime.timedelta(days=3650))
    assert guard.remaining_timeout is None