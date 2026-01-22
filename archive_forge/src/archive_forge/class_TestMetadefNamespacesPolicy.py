from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
class TestMetadefNamespacesPolicy(functional.SynchronousAPIBase):

    def setUp(self):
        super(TestMetadefNamespacesPolicy, self).setUp()
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)

    def set_policy_rules(self, rules):
        self.policy.set_rules(oslo_policy.policy.Rules.from_dict(rules), overwrite=True)

    def start_server(self):
        with mock.patch.object(policy, 'Enforcer') as mock_enf:
            mock_enf.return_value = self.policy
            super(TestMetadefNamespacesPolicy, self).start_server()

    def _verify_forbidden_converted_to_not_found(self, path, method, json=None):
        headers = self._headers({'X-Tenant-Id': 'fake-tenant-id', 'X-Roles': 'member'})
        resp = self.api_request(method, path, headers=headers, json=json)
        self.assertEqual(404, resp.status_code)

    def test_namespace_list_basic(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        path = '/v2/metadefs/namespaces'
        NAME_SPACE2['visibility'] = 'public'
        md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE2)
        self.assertEqual('MySecondNamespace', md_resource['namespace'])
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(2, len(md_resource['namespaces']))
        self.set_policy_rules({'get_metadef_namespaces': '!', 'get_metadef_namespace': '@'})
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)

    def test_namespace_list_with_resource_types(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path=path, data=GLOBAL_NAMESPACE_DATA)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(1, len(md_resource['namespaces']))
        for namespace_obj in md_resource['namespaces']:
            self.assertIn('resource_type_associations', namespace_obj)
        self.set_policy_rules({'get_metadef_namespaces': '@', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '!'})
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_namespaces': '@', 'get_metadef_namespace': '!', 'list_metadef_resource_types': '@'})
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual(0, len(md_resource['namespaces']))
        for namespace_obj in md_resource['namespaces']:
            self.assertNotIn('resource_type_associations', namespace_obj)

    def test_namespace_create_basic(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.set_policy_rules({'add_metadef_namespace': '!', 'get_metadef_namespace': '@'})
        resp = self.api_post(path, json=NAME_SPACE2)
        self.assertEqual(403, resp.status_code)

    def test_namespace_create_with_resource_type_associations(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        data = {'resource_type_associations': [{'name': 'MyResourceType', 'prefix': 'prefix_', 'properties_target': 'temp'}]}
        data.update(NAME_SPACE1)
        md_resource = self._create_metadef_resource(path=path, data=data)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.assertEqual('MyResourceType', md_resource['resource_type_associations'][0]['name'])
        self.set_policy_rules({'add_metadef_resource_type_association': '!', 'get_metadef_namespace': '@'})
        data.update(NAME_SPACE2)
        resp = self.api_post(path, json=data)
        self.assertEqual(403, resp.status_code)

    def test_namespace_create_with_objects(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        data = {'objects': [{'name': 'MyObject', 'description': 'My object for My namespace', 'properties': {'test_property': {'title': 'test_property', 'description': 'Test property for My object', 'type': 'string'}}}]}
        data.update(NAME_SPACE1)
        md_resource = self._create_metadef_resource(path=path, data=data)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.assertEqual('MyObject', md_resource['objects'][0]['name'])
        self.set_policy_rules({'add_metadef_object': '!', 'get_metadef_namespace': '@'})
        data.update(NAME_SPACE2)
        resp = self.api_post(path, json=data)
        self.assertEqual(403, resp.status_code)

    def test_namespace_create_with_tags(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        data = {'tags': [{'name': 'MyTag'}]}
        data.update(NAME_SPACE1)
        md_resource = self._create_metadef_resource(path=path, data=data)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.assertEqual('MyTag', md_resource['tags'][0]['name'])
        data.update(NAME_SPACE2)
        self.set_policy_rules({'add_metadef_tag': '!', 'get_metadef_namespace': '@'})
        resp = self.api_post(path, json=data)
        self.assertEqual(403, resp.status_code)

    def test_namespace_create_with_properties(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        data = {'properties': {'TestProperty': {'title': 'MyTestProperty', 'description': 'Test Property for My namespace', 'type': 'string'}}}
        data.update(NAME_SPACE1)
        md_resource = self._create_metadef_resource(path=path, data=data)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.assertEqual('MyTestProperty', md_resource['properties']['TestProperty']['title'])
        data.update(NAME_SPACE2)
        self.set_policy_rules({'add_metadef_property': '!', 'get_metadef_namespace': '@'})
        resp = self.api_post(path, json=data)
        self.assertEqual(403, resp.status_code)

    def test_namespace_get_basic(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path=path, data=GLOBAL_NAMESPACE_DATA)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.assertIn('objects', md_resource)
        self.assertIn('resource_type_associations', md_resource)
        self.assertIn('tags', md_resource)
        self.assertIn('properties', md_resource)
        self.set_policy_rules({'get_metadef_namespace': '!'})
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'GET')
        self.set_policy_rules({'get_metadef_objects': '!', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '@', 'get_metadef_properties': '@', 'get_metadef_tags': '@'})
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_objects': '@', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '!', 'get_metadef_properties': '@', 'get_metadef_tags': '@'})
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_objects': '@', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '@', 'get_metadef_properties': '!', 'get_metadef_tags': '@'})
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'get_metadef_objects': '@', 'get_metadef_namespace': '@', 'list_metadef_resource_types': '@', 'get_metadef_properties': '@', 'get_metadef_tags': '!'})
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_get(path)
        self.assertEqual(403, resp.status_code)

    def test_namespace_update_basic(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path=path, data=NAME_SPACE1)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.assertEqual('private', md_resource['visibility'])
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        data = {'visibility': 'public', 'namespace': md_resource['namespace']}
        resp = self.api_put(path, json=data)
        md_resource = resp.json
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.assertEqual('public', md_resource['visibility'])
        self.set_policy_rules({'modify_metadef_namespace': '!', 'get_metadef_namespace': '@'})
        resp = self.api_put(path, json=data)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'modify_metadef_namespace': '@', 'get_metadef_namespace': '@'})
        path = '/v2/metadefs/namespaces/non-existing'
        resp = self.api_put(path, json=data)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'modify_metadef_namespace': '!', 'get_metadef_namespace': '!'})
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_put(path, json=data)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'modify_metadef_namespace': '@', 'get_metadef_namespace': '@'})
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        data = {'visibility': 'private', 'namespace': md_resource['namespace']}
        resp = self.api_put(path, json=data)
        md_resource = resp.json
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.assertEqual('private', md_resource['visibility'])
        self._verify_forbidden_converted_to_not_found(path, 'PUT', json=data)

    def test_namespace_delete_basic(self):

        def _create_private_namespace(fn_call, data):
            path = '/v2/metadefs/namespaces'
            return fn_call(path=path, data=data)
        self.start_server()
        md_resource = _create_private_namespace(self._create_metadef_resource, NAME_SPACE1)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_delete(path)
        self.assertEqual(204, resp.status_code)
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_get(path)
        self.assertEqual(404, resp.status_code)
        md_resource = _create_private_namespace(self._create_metadef_resource, NAME_SPACE2)
        self.assertEqual('MySecondNamespace', md_resource['namespace'])
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
        path = '/v2/metadefs/namespaces/non-existing'
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '!'})
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'DELETE')

    def test_namespace_delete_objects_basic(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path, data=GLOBAL_NAMESPACE_DATA)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        self.assertIn('objects', md_resource)
        path = '/v2/metadefs/namespaces/%s/objects' % md_resource['namespace']
        resp = self.api_delete(path)
        self.assertEqual(204, resp.status_code)
        path = '/v2/metadefs/namespaces/%s' % md_resource['namespace']
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertNotIn('objects', md_resource)
        self.assertEqual('MyNamespace', md_resource['namespace'])
        path = '/v2/metadefs/namespaces/%s/objects' % md_resource['namespace']
        data = {'name': 'MyObject', 'description': 'My object for My namespace', 'properties': {'test_property': {'title': 'test_property', 'description': 'Test property for My object', 'type': 'string'}}}
        md_object = self._create_metadef_resource(path, data=data)
        self.assertEqual('MyObject', md_object['name'])
        path = '/v2/metadefs/namespaces/%s/objects' % md_resource['namespace']
        self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        path = '/v2/metadefs/namespaces/non-existing/objects'
        self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metaded_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '!'})
        path = '/v2/metadefs/namespaces/%s/objects' % md_resource['namespace']
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'DELETE')

    def test_namespace_delete_properties_basic(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path, data=GLOBAL_NAMESPACE_DATA)
        namespace = md_resource['namespace']
        self.assertEqual('MyNamespace', namespace)
        self.assertIn('properties', md_resource)
        path = '/v2/metadefs/namespaces/%s/properties' % namespace
        resp = self.api_delete(path)
        self.assertEqual(204, resp.status_code)
        path = '/v2/metadefs/namespaces/%s' % namespace
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertNotIn('properties', md_resource)
        self.assertEqual('MyNamespace', namespace)
        path = '/v2/metadefs/namespaces/%s/properties' % namespace
        data = {'name': 'MyProperty', 'title': 'test_property', 'description': 'Test property for My Namespace', 'type': 'string'}
        md_resource = self._create_metadef_resource(path, data=data)
        self.assertEqual('MyProperty', md_resource['name'])
        path = '/v2/metadefs/namespaces/%s/properties' % namespace
        self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        path = '/v2/metadefs/namespaces/non-existing/properties'
        self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '!'})
        path = '/v2/metadefs/namespaces/%s/properties' % namespace
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'DELETE')

    def test_namespace_delete_tags_basic(self):
        self.start_server()
        path = '/v2/metadefs/namespaces'
        md_resource = self._create_metadef_resource(path, data=GLOBAL_NAMESPACE_DATA)
        namespace = md_resource['namespace']
        self.assertEqual('MyNamespace', namespace)
        self.assertIn('tags', md_resource)
        path = '/v2/metadefs/namespaces/%s/tags' % namespace
        resp = self.api_delete(path)
        self.assertEqual(204, resp.status_code)
        path = '/v2/metadefs/namespaces/%s' % namespace
        resp = self.api_get(path)
        md_resource = resp.json
        self.assertNotIn('tags', md_resource)
        self.assertEqual('MyNamespace', namespace)
        tag_name = 'MyTag'
        path = '/v2/metadefs/namespaces/%s/tags/%s' % (namespace, tag_name)
        md_resource = self._create_metadef_resource(path)
        self.assertEqual('MyTag', md_resource['name'])
        path = '/v2/metadefs/namespaces/%s/tags' % namespace
        self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        path = '/v2/metadefs/namespaces/%s/tags' % namespace
        self.set_policy_rules({'delete_metadef_namespace': '@', 'delete_metadef_tags': '!', 'get_metadef_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(403, resp.status_code)
        path = '/v2/metadefs/namespaces/non-existing/tags'
        self.set_policy_rules({'delete_metadef_namespace': '@', 'delete_metadef_tags': '@', 'get_metadef_namespace': '@'})
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_namespace': '!', 'get_metadef_namespace': '!', 'delete_metadef_tags': '!'})
        path = '/v2/metadefs/namespaces/%s/tags' % namespace
        resp = self.api_delete(path)
        self.assertEqual(404, resp.status_code)
        self.set_policy_rules({'delete_metadef_namespace': '@', 'get_metadef_namespace': '@', 'delete_metadef_tags': '@'})
        self._verify_forbidden_converted_to_not_found(path, 'DELETE')