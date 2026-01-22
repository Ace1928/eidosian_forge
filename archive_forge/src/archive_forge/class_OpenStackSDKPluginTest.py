from unittest import mock
from openstack import exceptions
from heat.engine.clients.os import openstacksdk
from heat.tests import common
from heat.tests import utils
class OpenStackSDKPluginTest(common.HeatTestCase):

    @mock.patch('openstack.connection.Connection')
    def setUp(self, mock_connection):
        super(OpenStackSDKPluginTest, self).setUp()
        context = utils.dummy_context()
        self.plugin = context.clients.client_plugin('openstack')

    def test_create(self):
        client = self.plugin.client()
        self.assertIsNotNone(client.network.segments)

    def test_is_not_found(self):
        self.assertFalse(self.plugin.is_not_found(exceptions.HttpException(http_status=400)))
        self.assertFalse(self.plugin.is_not_found(Exception))
        self.assertTrue(self.plugin.is_not_found(exceptions.NotFoundException(http_status=404)))