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
def test_public_version_v3(self):
    client = TestClient(self.public_app)
    resp = client.get('/v3/')
    self.assertEqual(http.client.OK, resp.status_int)
    data = jsonutils.loads(resp.body)
    expected = v3_VERSION_RESPONSE
    self._paste_in_port(expected['version'], 'http://localhost:%s/v3/' % self.public_port)
    self.assertEqual(expected, data)