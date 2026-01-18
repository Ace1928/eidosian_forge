from saharaclient.api import plugins
from saharaclient.tests.unit import base
def test_plugins_get_version_details(self):
    url = self.URL + '/plugins/name/1'
    self.responses.get(url, json={'plugin': self.body})
    resp = self.client.plugins.get_version_details('name', '1')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, plugins.Plugin)
    self.assertFields(self.body, resp)