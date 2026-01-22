from unittest import mock
from keystoneauth1 import exceptions as keystone_exceptions
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.tests import common
class KeystoneClientPluginProjectTest(common.HeatTestCase):
    sample_uuid = '477e8273-60a7-4c41-b683-fdb0bc7cd152'
    sample_name = 'sample_project'
    sample_name_and_domain = 'sample_project{sample_domain}'
    sample_domain_uuid = '577e8273-60a7-4c41-b683-fdb0bc7cd152'
    sample_domain_name = 'sample_domain'
    sample_name_and_domain_invalid_input = 'sample_project@@'

    def _get_mock_project(self):
        project = mock.MagicMock()
        project.id = self.sample_uuid
        project.name = self.sample_name
        project.name_and_domain = self.sample_name_and_domain
        return project

    def setUp(self):
        super(KeystoneClientPluginProjectTest, self).setUp()
        self._client = mock.MagicMock()

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_project_id(self, client_keystone):
        self._client.client.projects.get.return_value = self._get_mock_project()
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        self.assertEqual(self.sample_uuid, client_plugin.get_project_id(self.sample_uuid))
        self._client.client.projects.get.assert_called_once_with(self.sample_uuid)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_project_id_with_name(self, client_keystone):
        self._client.client.projects.get.side_effect = keystone_exceptions.NotFound
        self._client.client.projects.list.return_value = [self._get_mock_project()]
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        self.assertEqual(self.sample_uuid, client_plugin.get_project_id(self.sample_name))
        self.assertRaises(keystone_exceptions.NotFound, self._client.client.projects.get, self.sample_name)
        self._client.client.projects.list.assert_called_once_with(domain=None, name=self.sample_name)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_project_id_with_name_and_domain(self, client_keystone):
        self._client.client.projects.get.side_effect = keystone_exceptions.NotFound
        self._client.client.projects.list.return_value = [self._get_mock_project()]
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        self.assertEqual(self.sample_uuid, client_plugin.get_project_id(self.sample_name_and_domain))
        self.assertRaises(keystone_exceptions.NotFound, self._client.client.projects.get, self.sample_name)
        self._client.client.projects.list.assert_called_once_with(domain=client_plugin.get_domain_id(self.sample_domain_uuid), name=self.sample_name)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_project_id_not_found(self, client_keystone):
        self._client.client.projects.get.side_effect = keystone_exceptions.NotFound
        self._client.client.projects.list.return_value = []
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        ex = self.assertRaises(exception.EntityNotFound, client_plugin.get_project_id, self.sample_name)
        msg = 'The KeystoneProject (%(name)s) could not be found.' % {'name': self.sample_name}
        self.assertEqual(msg, str(ex))
        self.assertRaises(keystone_exceptions.NotFound, self._client.client.projects.get, self.sample_name)
        self._client.client.projects.list.assert_called_once_with(domain=None, name=self.sample_name)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_project_id_with_domain_not_found(self, client_keystone):
        self._client.client.projects.get.side_effect = keystone_exceptions.NotFound
        self._client.client.projects.list.return_value = []
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        ex = self.assertRaises(exception.EntityNotFound, client_plugin.get_project_id, self.sample_name_and_domain)
        msg = 'The KeystoneProject (%(name)s) could not be found.' % {'name': self.sample_name}
        self.assertEqual(msg, str(ex))
        self.assertRaises(keystone_exceptions.NotFound, self._client.client.projects.get, self.sample_name)
        self._client.client.projects.list.assert_called_once_with(domain=client_plugin.get_domain_id(self.sample_domain_uuid), name=self.sample_name)

    @mock.patch.object(keystone.KeystoneClientPlugin, 'client')
    def test_get_project_id_with_name_and_domain_invalid_input(self, client_keystone):
        self._client.client.projects.get.side_effect = keystone_exceptions.NotFound
        self._client.client.projects.list.return_value = []
        client_keystone.return_value = self._client
        client_plugin = keystone.KeystoneClientPlugin(context=mock.MagicMock())
        self.assertRaises(exception.EntityNotFound, client_plugin.get_project_id, self.sample_name_and_domain_invalid_input)