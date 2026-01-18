import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_reauth_not_called(self):
    auth = CalledAuthPlugin(invalidate=True)
    sess = client_session.Session(auth=auth)
    self.requests_mock.get(self.TEST_URL, [{'text': 'Failed', 'status_code': 401}, {'text': 'Hello', 'status_code': 200}])
    self.assertRaises(exceptions.Unauthorized, sess.get, self.TEST_URL, authenticated=True, allow_reauth=False)
    self.assertFalse(auth.invalidate_called)