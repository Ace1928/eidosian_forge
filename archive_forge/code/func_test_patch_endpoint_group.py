import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_patch_endpoint_group(self):
    """PATCH /OS-EP-FILTER/endpoint_groups/{endpoint_group}.

        Valid endpoint group patch test case.

        """
    body = copy.deepcopy(self.DEFAULT_ENDPOINT_GROUP_BODY)
    body['endpoint_group']['filters'] = {'region_id': 'UK'}
    body['endpoint_group']['name'] = 'patch_test'
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s' % {'endpoint_group_id': endpoint_group_id}
    r = self.patch(url, body=body)
    self.assertEqual(endpoint_group_id, r.result['endpoint_group']['id'])
    self.assertEqual(body['endpoint_group']['filters'], r.result['endpoint_group']['filters'])
    self.assertThat(r.result['endpoint_group']['links']['self'], matchers.EndsWith(url))