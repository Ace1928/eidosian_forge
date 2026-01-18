import copy
import functools
import random
import http.client
from oslo_serialization import jsonutils
from testtools import matchers as tt_matchers
import webob
from keystone.api import discovery
from keystone.common import json_home
from keystone.tests import unit
def test_v2_disabled(self):
    client = TestClient(self.public_app)
    resp = client.get('/v2.0/')
    self.assertEqual(418, resp.status_int)
    resp = client.get('/v3/')
    self.assertEqual(http.client.OK, resp.status_int)
    data = jsonutils.loads(resp.body)
    expected = v3_VERSION_RESPONSE
    self._paste_in_port(expected['version'], 'http://localhost:%s/v3/' % self.public_port)
    self.assertEqual(expected, data)
    v3_only_response = {'versions': {'values': [v3_EXPECTED_RESPONSE]}}
    self._paste_in_port(v3_only_response['versions']['values'][0], 'http://localhost:%s/v3/' % self.public_port)
    resp = client.get('/')
    self.assertEqual(300, resp.status_int)
    data = jsonutils.loads(resp.body)
    self.assertEqual(v3_only_response, data)