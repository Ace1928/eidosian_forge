from heat.tests import common
from heat.tests import utils
class VitrageClientPluginTest(common.HeatTestCase):

    def test_create(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('vitrage')
        client = plugin.client()
        self.assertIsNotNone(client.template.list)