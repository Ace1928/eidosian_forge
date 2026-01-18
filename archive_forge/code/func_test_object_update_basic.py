from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_object_update_basic(self):
    self.start_server()
    self.load_data(create_objects=True)
    path = '/v2/metadefs/namespaces/%s/objects/%s' % (NAME_SPACE1['namespace'], OBJECT1['name'])
    data = {'name': OBJECT1['name'], 'description': 'My updated description'}
    resp = self.api_put(path, json=data)
    md_resource = resp.json
    self.assertEqual(data['description'], md_resource['description'])
    data = {'name': OBJECT2['name'], 'description': 'My updated description'}
    path = '/v2/metadefs/namespaces/%s/objects/%s' % (NAME_SPACE1['namespace'], OBJECT2['name'])
    self.set_policy_rules({'modify_metadef_object': '!', 'get_metadef_namespace': '@'})
    resp = self.api_put(path, json=data)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'modify_metadef_object': '!', 'get_metadef_namespace': '!'})
    resp = self.api_put(path, json=data)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'modify_metadef_object': '@', 'get_metadef_namespace': '@'})
    self._verify_forbidden_converted_to_not_found(path, 'PUT', json=data)