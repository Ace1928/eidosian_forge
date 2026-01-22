import collections
from unittest import mock
from barbicanclient import exceptions
from heat.common import exception
from heat.engine.clients.os import barbican
from heat.tests import common
from heat.tests import utils
class BarbicanClientPluginTest(common.HeatTestCase):

    def setUp(self):
        super(BarbicanClientPluginTest, self).setUp()
        self.barbican_client = mock.MagicMock()
        con = utils.dummy_context()
        c = con.clients
        self.barbican_plugin = c.client_plugin('barbican')
        self.barbican_plugin.client = lambda: self.barbican_client

    @mock.patch('keystoneauth1.discover.get_version_data', mock.MagicMock(return_value=[{'status': 'STABLE'}]))
    def test_create(self):
        context = utils.dummy_context()
        plugin = context.clients.client_plugin('barbican')
        client = plugin.client()
        self.assertIsNotNone(client.orders)

    def test_get_secret_by_ref(self):
        secret = collections.namedtuple('Secret', ['name'])('foo')
        self.barbican_client.secrets.get.return_value = secret
        self.assertEqual(secret, self.barbican_plugin.get_secret_by_ref('secret'))

    def test_get_secret_payload_by_ref(self):
        payload_content = 'payload content'
        secret = collections.namedtuple('Secret', ['name', 'payload'])('foo', payload_content)
        self.barbican_client.secrets.get.return_value = secret
        expect = payload_content
        self.assertEqual(expect, self.barbican_plugin.get_secret_payload_by_ref('secret'))

    def test_get_secret_payload_by_ref_not_found(self):
        exc = exceptions.HTTPClientError(message='Not Found', status_code=404)
        self.barbican_client.secrets.get.side_effect = exc
        self.assertRaises(exception.EntityNotFound, self.barbican_plugin.get_secret_payload_by_ref, 'secret')

    def test_get_secret_by_ref_not_found(self):
        exc = exceptions.HTTPClientError(message='Not Found', status_code=404)
        self.barbican_client.secrets.get.side_effect = exc
        self.assertRaises(exception.EntityNotFound, self.barbican_plugin.get_secret_by_ref, 'secret')