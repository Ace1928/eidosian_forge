from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_tags_delete_basic(self):
    self.start_server()
    self.load_data(create_tags=True)
    path = '/v2/metadefs/namespaces/%s/tags' % NAME_SPACE1['namespace']
    resp = self.api_delete(path)
    self.assertEqual(204, resp.status_code)
    path = '/v2/metadefs/namespaces/%s/tags' % NAME_SPACE1['namespace']
    resp = self.api_get(path)
    md_resource = resp.json
    self.assertEqual(0, len(md_resource['tags']))
    path = '/v2/metadefs/namespaces/%s/tags' % NAME_SPACE1['namespace']
    self.set_policy_rules({'delete_metadef_tags': '!', 'get_metadef_namespace': ''})
    resp = self.api_delete(path)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'delete_metadef_tags': '!', 'get_metadef_namespace': '!'})
    resp = self.api_delete(path)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'delete_metadef_tags': '', 'get_metadef_namespace': ''})
    self._verify_forbidden_converted_to_not_found(path, 'DELETE')