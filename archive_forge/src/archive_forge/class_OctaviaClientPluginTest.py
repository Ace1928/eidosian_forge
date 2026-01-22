from heat.tests import common
from heat.tests import utils
class OctaviaClientPluginTest(common.HeatTestCase):

    def test_create(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('octavia')
        client = plugin.client()
        self.assertIsNotNone(client.endpoint)