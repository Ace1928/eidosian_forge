from saharaclient.api import plugins
from saharaclient.tests.unit import base
def test_plugins_list(self):
    url = self.URL + '/plugins'
    self.responses.get(url, json={'plugins': [self.body]})
    resp = self.client.plugins.list()
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp[0], plugins.Plugin)
    self.assertFields(self.body, resp[0])