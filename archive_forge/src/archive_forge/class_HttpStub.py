import os
import sys
import mock
import OpenSSL
import pytest  # type: ignore
from six.moves import http_client
import urllib3  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._mtls_helper
import google.auth.transport.urllib3
from google.oauth2 import service_account
from tests.transport import compliance
class HttpStub(object):

    def __init__(self, responses, headers=None):
        self.responses = responses
        self.requests = []
        self.headers = headers or {}

    def urlopen(self, method, url, body=None, headers=None, **kwargs):
        self.requests.append((method, url, body, headers, kwargs))
        return self.responses.pop(0)

    def clear(self):
        pass