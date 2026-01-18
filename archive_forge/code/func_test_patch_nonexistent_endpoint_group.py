import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_patch_nonexistent_endpoint_group(self):
    """PATCH /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Invalid endpoint group patch test case.

        """
    body = {'endpoint_group': {'name': 'patch_test'}}
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': 'ABC'}
    self.patch(url, body=body, expected_status=http.client.NOT_FOUND)