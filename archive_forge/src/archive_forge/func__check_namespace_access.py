import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _check_namespace_access(namespaces, tenant):
    headers = self._headers({'X-Tenant-Id': tenant, 'X-Roles': 'reader,member'})
    for namespace in namespaces:
        path = self._url('/v2/metadefs/namespaces/%s' % namespace['namespace'])
        headers = headers
        response = requests.get(path, headers=headers)
        if namespace['visibility'] == 'public':
            self.assertEqual(http.OK, response.status_code)
        else:
            self.assertEqual(http.NOT_FOUND, response.status_code)