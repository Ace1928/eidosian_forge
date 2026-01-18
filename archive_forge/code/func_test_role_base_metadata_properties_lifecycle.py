import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def test_role_base_metadata_properties_lifecycle(self):
    path = self._url('/v2/metadefs/namespaces')
    headers = self._headers({'content-type': 'application/json'})
    tenant1_namespaces = []
    tenant2_namespaces = []
    for tenant in [self.tenant1, self.tenant2]:
        headers['X-Tenant-Id'] = tenant
        for visibility in ['public', 'private']:
            namespace_data = {'namespace': '%s_%s_namespace' % (tenant, visibility), 'display_name': 'My User Friendly Namespace', 'description': 'My description', 'visibility': visibility, 'owner': tenant}
            namespace = self.create_namespace(path, headers, namespace_data)
            self.assertNamespacesEqual(namespace, namespace_data)
            if tenant == self.tenant1:
                tenant1_namespaces.append(namespace)
            else:
                tenant2_namespaces.append(namespace)
    tenant1_properties = self._create_properties(tenant1_namespaces)
    tenant2_properties = self._create_properties(tenant2_namespaces)

    def _check_properties_access(properties, tenant):
        headers = self._headers({'content-type': 'application/json', 'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
        for prop in properties:
            for namespace, property_name in prop.items():
                path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace, property_name))
                response = requests.get(path, headers=headers)
                if namespace.split('_')[1] == 'public':
                    expected = http.OK
                else:
                    expected = http.NOT_FOUND
                self.assertEqual(expected, response.status_code)
                path = self._url('/v2/metadefs/namespaces/%s/properties' % namespace)
                response = requests.get(path, headers=headers)
                self.assertEqual(expected, response.status_code)
                if expected == http.OK:
                    resp_props = response.json()['properties'].values()
                    self.assertEqual(sorted(prop.values()), sorted([x['name'] for x in resp_props]))
    _check_properties_access(tenant2_properties, self.tenant1)
    _check_properties_access(tenant1_properties, self.tenant2)
    total_properties = tenant1_properties + tenant2_properties
    for prop in total_properties:
        for namespace, property_name in prop.items():
            data = {'name': property_name, 'type': 'string', 'title': 'string property', 'description': 'desc-UPDATED'}
            path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace, property_name))
            headers['X-Roles'] = 'reader,member'
            response = requests.put(path, headers=headers, json=data)
            self.assertEqual(http.FORBIDDEN, response.status_code)
            headers = self._headers({'X-Tenant-Id': namespace.split('_')[0]})
            self._update_property(path, headers, data)
    for prop in total_properties:
        for namespace, property_name in prop.items():
            path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace, property_name))
            response = requests.delete(path, headers=self._headers({'X-Roles': 'reader,member', 'X-Tenant-Id': namespace.split('_')[0]}))
            self.assertEqual(http.FORBIDDEN, response.status_code)
    headers = self._headers()
    for prop in total_properties:
        for namespace, property_name in prop.items():
            path = self._url('/v2/metadefs/namespaces/%s/properties/%s' % (namespace, property_name))
            response = requests.delete(path, headers=headers)
            self.assertEqual(http.NO_CONTENT, response.status_code)
            response = requests.get(path, headers=headers)
            self.assertEqual(http.NOT_FOUND, response.status_code)