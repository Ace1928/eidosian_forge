from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_resourcetypes_list_basic(self):
    self.start_server()
    self.load_data(create_resourcetypes=True)
    path = '/v2/metadefs/resource_types'
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual(1, len(md_resource))
    self.set_policy_rules({'list_metadef_resource_types': '!', 'get_metadef_namespace': '@'})
    resp = self.api_get(path)
    self.assertEqual(403, resp.status_code)