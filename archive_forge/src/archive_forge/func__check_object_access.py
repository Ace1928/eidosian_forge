import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _check_object_access(objects, tenant):
    headers = self._headers({'content-type': 'application/json', 'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
    for obj in objects:
        for namespace, object_name in obj.items():
            path = self._url('/v2/metadefs/namespaces/%s/objects/%s' % (namespace, object_name))
            headers = headers
            response = requests.get(path, headers=headers)
            if namespace.split('_')[1] == 'public':
                expected = http.OK
            else:
                expected = http.NOT_FOUND
            self.assertEqual(expected, response.status_code)
            path = self._url('/v2/metadefs/namespaces/%s/objects' % namespace)
            response = requests.get(path, headers=headers)
            self.assertEqual(expected, response.status_code)
            if expected == http.OK:
                resp_objs = response.json()['objects']
                self.assertEqual(sorted(obj.values()), sorted([x['name'] for x in resp_objs]))