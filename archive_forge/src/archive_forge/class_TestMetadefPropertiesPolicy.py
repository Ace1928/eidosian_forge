from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
class TestMetadefPropertiesPolicy(functional.SynchronousAPIBase):

    def setUp(self):
        super(TestMetadefPropertiesPolicy, self).setUp()
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)

    def load_data(self, create_properties=False):
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        if create_properties:
            namespace = md_resource['namespace']
            path = '/v2/metadefs/namespaces/%s/properties' % namespace
            for prop in [PROPERTY1, PROPERTY2]:
                md_resource = self._create_metadef_resource(path=path, data=prop)
                self.assertEqual(prop['name'], md_resource['name'])

    def set_policy_rules(self, rules):
        self.policy.set_rules(oslo_policy.policy.Rules.from_dict(rules), overwrite=True)

    def start_server(self):
        with mock.patch.object(policy, 'Enforcer') as mock_enf:
            mock_enf.return_value = self.policy
            super(TestMetadefPropertiesPolicy, self).start_server()

    def _verify_forbidden_converted_to_not_found(self, path, method, json=None):
        headers = self._headers({'X-Tenant-Id': 'fake-tenant-id', 'X-Roles': 'member'})
        resp = self.api_request(method, path, headers=headers, json=json)
        self.assertEqual(404, resp.status_code)

    def test_property_list_basic(self):
        self.start_server()
        self.load_data(create_properties=True)
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/properties' % namespace
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(2, len(md_resource['properties']))
        self.set_policy_rules({'get_metadef_properties': '!', 'get_metadef_namespace': ''})
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_properties': '!', 'get_metadef_namespace': '!'})
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_metadef_properties': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'GET')

    def test_property_get_basic(self):
        self.start_server()
        self.load_data(create_properties=True)
        path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY1['name'])
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(PROPERTY1['name'], md_resource['name'])
        self.set_policy_rules({'get_metadef_property': '!', 'get_metadef_namespace': '', 'get_metadef_resource_type': ''})
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_property': '', 'get_metadef_namespace': '', 'get_metadef_resource_type': '!'})
        url_path = "%s?resource_type='abcd'" % path
        resp = self.api_get(url_path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_property': '!', 'get_metadef_namespace': '!', 'get_metadef_resource_type': '!'})
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_metadef_property': '', 'get_metadef_namespace': '', 'get_metadef_resource_type': ''})
        self._verify_forbidden_converted_to_not_found(path, 'GET')

    def test_property_create_basic(self):
        self.start_server()
        self.load_data()
        namespace = NAME_SPACE1['namespace']
        path = '/v2/metadefs/namespaces/%s/properties' % namespace
        md_resource = self._create_metadef_resource(path=path, data=PROPERTY1)
        self.assertEqual('MyProperty', md_resource['name'])
        self.set_policy_rules({'add_metadef_property': '!', 'get_metadef_namespace': ''})
        resp = self.api_post(path, json=PROPERTY2)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'add_metadef_property': '!', 'get_metadef_namespace': '!'})
        resp = self.api_post(path, json=PROPERTY2)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'add_metadef_property': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'POST', json=PROPERTY2)

    def test_property_update_basic(self):
        self.start_server()
        self.load_data(create_properties=True)
        path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY1['name'])
        data = {'name': PROPERTY1['name'], 'title': PROPERTY1['title'], 'type': PROPERTY1['type'], 'description': 'My updated description'}
        resp = self.api_put(path, json=data)
        md_resource = resp.json
        self.assertEqual(data['description'], md_resource['description'])
        data = {'name': PROPERTY2['name'], 'title': PROPERTY2['title'], 'type': PROPERTY2['type'], 'description': 'My updated description'}
        path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY2['name'])
        self.set_policy_rules({'modify_metadef_property': '!', 'get_metadef_namespace': ''})
        resp = self.api_put(path, json=data)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'modify_metadef_property': '!', 'get_metadef_namespace': '!'})
        resp = self.api_put(path, json=data)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'modify_metadef_property': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'PUT', json=data)

    def test_property_delete_basic(self):
        self.start_server()
        self.load_data(create_properties=True)
        path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY1['name'])
        resp = self.api_delete(path)
        self.assertEqual(204, resp.status_code)
        path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY1['name'])
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        path = '/v2/metadefs/namespaces/%s/properties/%s' % (NAME_SPACE1['namespace'], PROPERTY2['name'])
        self.set_policy_rules({'remove_metadef_property': '!', 'get_metadef_namespace': ''})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'remove_metadef_property': '!', 'get_metadef_namespace': '!'})
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'remove_metadef_property': '', 'get_metadef_namespace': ''})
        self._verify_forbidden_converted_to_not_found(path, 'DELETE')