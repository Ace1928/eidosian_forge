from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
class KeystoneClientPluginDomainTest(common.HeatTestCase):
    sample_uuid = '477e8273-60a7-4c41-b683-fdb0bc7cd152'
    sample_name = 'sample_domain'

    def _get_mock_domain(self):
        domain = mock.MagicMock()
        domain.id = self.sample_uuid
        domain.name = self.sample_name
        return domain

    def setUp(self):
        super(KeystoneClientPluginDomainTest, self).setUp()
        self._client = mock.MagicMock()

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_domain_id(self, client_keystone):
        self._client.client.domains.get.return_value = self._get_mock_domain()
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        self.assertEqual(self.sample_uuid, client_plugin.get_domain_id(self.sample_uuid))
        self._client.client.domains.get.assert_called_once_with(self.sample_uuid)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_domain_id_with_name(self, client_keystone):
        self._client.client.domains.get.side_effect = keystone_exceptions.NotFound
        self._client.client.domains.list.return_value = [self._get_mock_domain()]
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        self.assertEqual(self.sample_uuid, client_plugin.get_domain_id(self.sample_name))
        self.assertRaises(keystone_exceptions.NotFound, self._client.client.domains.get, self.sample_name)
        self._client.client.domains.list.assert_called_once_with(name=self.sample_name)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_domain_id_not_found(self, client_keystone):
        self._client.client.domains.get.side_effect = keystone_exceptions.NotFound
        self._client.client.domains.list.return_value = []
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        ex = self.assertRaises(exception.EntityNotFound, client_plugin.get_domain_id, self.sample_name)
        msg = 'The KeystoneDomain (%(name)s) could not be found.' % {'name': self.sample_name}
        self.assertEqual(msg, str(ex))
        self.assertRaises(keystone_exceptions.NotFound, self._client.client.domains.get, self.sample_name)
        self._client.client.domains.list.assert_called_once_with(name=self.sample_name)