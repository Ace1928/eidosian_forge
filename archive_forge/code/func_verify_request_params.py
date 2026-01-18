import datetime
import json
import mock
import pytest  # type: ignore
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import _jwt_async as jwt
from google.auth import exceptions
from google.oauth2 import _client as sync_client
from google.oauth2 import _client_async as _client
from tests.oauth2 import test__client as test_client
def verify_request_params(request, params):
    request_body = request.call_args[1]['body'].decode('utf-8')
    request_params = urllib.parse.parse_qs(request_body)
    for key, value in six.iteritems(params):
        assert request_params[key][0] == value