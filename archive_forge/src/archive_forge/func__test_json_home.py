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
def _test_json_home(self, path, exp_json_home_data):
    client = TestClient(self.public_app)
    resp = client.get(path, headers={'Accept': 'application/json-home'})
    self.assertThat(resp.status, tt_matchers.Equals('200 OK'))
    self.assertThat(resp.headers['Content-Type'], tt_matchers.Equals('application/json-home'))
    maxDiff = self.maxDiff
    self.maxDiff = None
    self.assertDictEqual(exp_json_home_data, jsonutils.loads(resp.body))
    self.maxDiff = maxDiff