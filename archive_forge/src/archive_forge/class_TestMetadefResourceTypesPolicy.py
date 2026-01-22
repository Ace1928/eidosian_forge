from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
class TestMetadefResourceTypesPolicy(functional.SynchronousAPIBase):

    def setUp(self):
        super(TestMetadefResourceTypesPolicy, self).setUp()
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)

    def load_data(self, create_resourcetypes=False):
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        if create_resourcetypes:
            namespace = md_resource['namespace']
            path = '/v2/metadefs/namespaces/%s/resource_types' % namespace
            for resource in [RESOURCETYPE_1, RESOURCETYPE_2]:
                md_resource = self._create_metadef_resource(path=path, data=resource)
                self.assertEqual(resource['name'], md_resource['name'])

    def set_policy_rules(self, rules):
        self.policy.set_rules(oslo_policy.policy.Rules.from_dict(rules), overwrite=True)

    def start_server(self):
        with mock.patch.object(policy, 'Enforcer') as mock_enf:
            mock_enf.return_value = self.policy
            super(TestMetadefResourceTypesPolicy, self).start_server()

    def _verify_forbidden_converted_to_not_found(self, path, method, json=None):
        headers = self._headers({'X-Tenant-Id': 'fake-tenant-id', 'X-Roles': 'member'})
        resp = self.api_request(method, path, headers=headers, json=json)
        self.assertEqual(404, resp.status_code)

    def test_namespace_resourcetypes_list_basic(self):
        self.start_server()
        self.load_data(create_resourcetypes=True)
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/resource_types' % namespace
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(2, len(md_resource['resource_type_associations']))
        self.set_policy_rules({'list_metadef_resource_types': '!', 'get_metadef_namespace': '@'})
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'list_metadef_resource_types': '!', 'get_metadef_namespace': '!'})
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'list_metadef_resource_types': '@', 'get_metadef_resource_type': '!', 'get_metadef_namespace': '@'})
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(0, len(md_resource['resource_type_associations']))
        self.set_policy_rules({'list_metadef_resource_types': '@', 'get_metadef_resource_type': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'GET')

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

    def test_resourcetype_create_basic(self):
        self.start_server()
        self.load_data()
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/resource_types' % namespace
        md_resource = self._create_metadef_resource(path=path, data=RESOURCETYPE_1)
        self.assertEqual('MyResourceType', md_resource['name'])
        self.set_policy_rules({'add_metadef_resource_type_association': '!', 'get_metadef_namespace': '@'})
        resp = self.api_post(path, json=RESOURCETYPE_2)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'add_metadef_resource_type_association': '!', 'get_metadef_namespace': '!'})
        resp = self.api_post(path, json=RESOURCETYPE_2)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'add_metadef_resource_type_association': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'POST', json=RESOURCETYPE_2)

    def test_object_delete_basic(self):
        self.start_server()
        self.load_data(create_resourcetypes=True)
        path = '/v2/metadefs/namespaces/%s/resource_types/%s' % (NAME_SPACE1['namespace'], RESOURCETYPE_1['name'])
        resp = self.api_delete(path)
        self.assertEqual(204, resp.status_code)
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/resource_types' % namespace
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(1, len(md_resource['resource_type_associations']))
        for resource in md_resource['resource_type_associations']:
            self.assertNotEqual(RESOURCETYPE_1['name'], resource['name'])
        path = '/v2/metadefs/namespaces/%s/resource_types/%s' % (NAME_SPACE1['namespace'], RESOURCETYPE_2['name'])
        self.set_policy_rules({'remove_metadef_resource_type_association': '!', 'get_metadef_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'remove_metadef_resource_type_association': '!', 'get_metadef_namespace': '!'})
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'remove_metadef_resource_type_association': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'DELETE')