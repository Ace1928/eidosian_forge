import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
def test_update_endpoint_enabled_false(self):
    """Call ``PATCH /endpoints/{endpoint_id}`` with enabled: False."""
    r = self.patch('/endpoints/%(endpoint_id)s' % {'endpoint_id': self.endpoint_id}, body={'endpoint': {'enabled': False}})
    exp_endpoint = copy.copy(self.endpoint)
    exp_endpoint['enabled'] = False
    self.assertValidEndpointResponse(r, exp_endpoint)