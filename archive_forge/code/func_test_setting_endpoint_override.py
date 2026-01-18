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
def test_setting_endpoint_override(self):
    endpoint_override = 'http://overrideurl'
    path = '/path'
    endpoint_url = endpoint_override + path
    auth = CalledAuthPlugin()
    sess = client_session.Session(auth=auth)
    adpt = adapter.Adapter(sess, endpoint_override=endpoint_override)
    response = uuid.uuid4().hex
    self.requests_mock.get(endpoint_url, text=response)
    resp = adpt.get(path)
    self.assertEqual(response, resp.text)
    self.assertEqual(endpoint_url, self.requests_mock.last_request.url)
    self.assertEqual(endpoint_override, adpt.get_endpoint())