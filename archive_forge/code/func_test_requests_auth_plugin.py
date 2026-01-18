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
def test_requests_auth_plugin(self):
    sess = client_session.Session()
    requests_auth = RequestsAuth()
    self.requests_mock.get(self.TEST_URL, text='resp')
    sess.get(self.TEST_URL, requests_auth=requests_auth)
    last = self.requests_mock.last_request
    self.assertEqual(requests_auth.header_val, last.headers[requests_auth.header_name])
    self.assertTrue(requests_auth.called)