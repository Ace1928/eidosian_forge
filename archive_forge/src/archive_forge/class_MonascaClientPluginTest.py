from unittest import mock
from heat.common import exception as heat_exception
from heat.engine.clients.os import monasca as client_plugin
from heat.tests import common
from heat.tests import utils
class MonascaClientPluginTest(common.HeatTestCase):

    def test_client(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('monasca')
        client = plugin.client()
        self.assertIsNotNone(client.metrics)

    def test_client_uses_session(self):
        context = mock.MagicMock()
        monasca_client = client_plugin.MonascaClientPlugin(context=context)
        self.assertIsNotNone(monasca_client._create())