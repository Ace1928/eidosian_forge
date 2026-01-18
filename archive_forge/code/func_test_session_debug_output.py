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
def test_session_debug_output(self):
    """Test request and response headers in debug logs.

        in order to redact secure headers while debug is true.
        """
    session = client_session.Session(verify=False)
    headers = {'HEADERA': 'HEADERVALB', 'Content-Type': 'application/json'}
    security_headers = {'Authorization': uuid.uuid4().hex, 'X-Auth-Token': uuid.uuid4().hex, 'X-Subject-Token': uuid.uuid4().hex, 'X-Service-Token': uuid.uuid4().hex}
    body = '{"a": "b"}'
    data = '{"c": "d"}'
    all_headers = dict(itertools.chain(headers.items(), security_headers.items()))
    self.stub_url('POST', text=body, headers=all_headers)
    resp = session.post(self.TEST_URL, headers=all_headers, data=data)
    self.assertEqual(resp.status_code, 200)
    self.assertIn('curl', self.logger.output)
    self.assertIn('POST', self.logger.output)
    self.assertIn('--insecure', self.logger.output)
    self.assertIn(body, self.logger.output)
    self.assertIn("'%s'" % data, self.logger.output)
    for k, v in headers.items():
        self.assertIn(k, self.logger.output)
        self.assertIn(v, self.logger.output)
    for k, v in security_headers.items():
        self.assertIn('%s: {SHA256}' % k, self.logger.output)
        self.assertEqual(v, resp.headers[k])
        self.assertNotIn(v, self.logger.output)