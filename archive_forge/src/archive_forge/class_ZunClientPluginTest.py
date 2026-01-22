from unittest import mock
from heat.tests import common
from heat.tests import utils
class ZunClientPluginTest(common.HeatTestCase):

    def setUp(self):
        super(ZunClientPluginTest, self).setUp()
        self.client = mock.Mock()
        context = utils.dummy_context()
        self.plugin = context.clients.client_plugin('zun')
        self.plugin.client = lambda **kw: self.client
        self.resource_id = '123456'

    def test_create(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('zun')
        client = plugin.client()
        self.assertEqual('http://server.test:5000/v3', client.containers.api.session.auth.endpoint)
        self.assertEqual('1.12', client.api_version.get_string())

    def test_container_update(self):
        prop_diff = {'cpu': 10, 'memory': 10, 'name': 'fake-container'}
        self.plugin.update_container(self.resource_id, **prop_diff)
        self.client.containers.update.assert_called_once_with(self.resource_id, cpu=10, memory=10, name='fake-container')