from unittest import mock
from heat.tests import common
from heat.tests import utils
class BlazarClientPluginTest(common.HeatTestCase):

    def setUp(self):
        super(BlazarClientPluginTest, self).setUp()
        self.blazar_client = mock.MagicMock()
        context = utils.dummy_context()
        self.blazar_client_plugin = context.clients.client_plugin('blazar')

    def _stub_client(self):
        self.blazar_client_plugin.client = lambda: self.blazar_client

    def test_create(self):
        client = self.blazar_client_plugin.client()
        self.assertEqual(None, client.blazar_url)

    def test_has_host_pass(self):
        self._stub_client()
        self.blazar_client.host.list.return_value = ['hosta']
        self.assertEqual(True, self.blazar_client_plugin.has_host())

    def test_has_host_fail(self):
        self._stub_client()
        self.blazar_client.host.list.return_value = []
        self.assertEqual(False, self.blazar_client_plugin.has_host())