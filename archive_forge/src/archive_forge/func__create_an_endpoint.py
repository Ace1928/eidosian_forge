import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _create_an_endpoint():
    endpoint_ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id)
    r = self.post('/endpoints', body={'endpoint': endpoint_ref})
    return r.result['endpoint']['id']