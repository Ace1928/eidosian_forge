import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _update_tags(self, path, headers, data):
    response = requests.put(path, headers=headers, json=data)
    self.assertEqual(http.OK, response.status_code, response.text)
    metadata_tag = response.json()
    self.assertEqual(data['name'], metadata_tag['name'])
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    self.assertEqual(data['name'], metadata_tag['name'])