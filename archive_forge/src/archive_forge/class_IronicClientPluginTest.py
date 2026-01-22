from unittest import mock
from ironicclient import exceptions as ic_exc
from heat.engine.clients.os import ironic as ic
from heat.tests import common
from heat.tests import utils
class IronicClientPluginTest(common.HeatTestCase):

    def test_create(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('ironic')
        client = plugin.client()
        self.assertEqual('http://server.test:5000/v3', client.port.api.session.auth.endpoint)