import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _update_object(self, path, headers, data, namespace):
    response = requests.put(path, headers=headers, json=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    expected_object = {'description': data['description'], 'name': data['name'], 'properties': data['properties'], 'required': data['required'], 'schema': '/v2/schemas/metadefs/object', 'self': '/v2/metadefs/namespaces/%s/objects/%s' % (namespace, data['name'])}
    metadata_object = response.json()
    metadata_object.pop('created_at')
    metadata_object.pop('updated_at')
    self.assertEqual(metadata_object, expected_object)
    response = requests.get(path, headers=self._headers())
    metadata_object = response.json()
    metadata_object.pop('created_at')
    metadata_object.pop('updated_at')
    self.assertEqual(metadata_object, expected_object)