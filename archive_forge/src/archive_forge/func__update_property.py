import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _update_property(self, path, headers, data):
    response = requests.put(path, headers=headers, json=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    property_object = response.json()
    self.assertEqual('string', property_object['type'])
    self.assertEqual(data['description'], property_object['description'])
    response = requests.get(path, headers=self._headers())
    self.assertEqual('string', property_object['type'])
    self.assertEqual(data['description'], property_object['description'])