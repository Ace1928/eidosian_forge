import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_patch_invalid_endpoint_group(self):
    """PATCH /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Valid endpoint group patch test case.

        """
    body = {'endpoint_group': {'description': 'endpoint group description', 'filters': {'region': 'UK'}, 'name': 'patch_test'}}
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
    self.patch(url, body=body, expected_status=http.client.BAD_REQUEST)
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
    r = self.get(url)
    del r.result['endpoint_group']['id']
    del r.result['endpoint_group']['links']
    self.assertDictEqual(self.DEFAULT_ENDPOINT_GROUP_BODY, r.result)