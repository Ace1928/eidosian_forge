from unittest import mock
import fixtures
from urllib import parse as urlparse
import uuid
from testtools import matchers
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient import utils as client_utils
from keystoneclient.v3.contrib.oauth1 import access_tokens
from keystoneclient.v3.contrib.oauth1 import auth
from keystoneclient.v3.contrib.oauth1 import consumers
from keystoneclient.v3.contrib.oauth1 import request_tokens
def _new_oauth_token_with_expires_at(self):
    key, secret, token = self._new_oauth_token()
    expires_at = client_utils.strtime()
    params = {'oauth_token': key, 'oauth_token_secret': secret, 'oauth_expires_at': expires_at}
    token = urlparse.urlencode(params)
    return (key, secret, expires_at, token)