import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _create_valid_endpoint_group(self, url, body):
    r = self.post(url, body=body)
    return r.result['endpoint_group']['id']