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
class TimeTickAdapterStub(AdapterStub):
    """Adapter that spends some (mocked) time when making a request."""

    def __init__(self, time_tick, responses, headers=None):
        self._time_tick = time_tick
        super(TimeTickAdapterStub, self).__init__(responses, headers=headers)

    def send(self, request, **kwargs):
        self._time_tick()
        return super(TimeTickAdapterStub, self).send(request, **kwargs)