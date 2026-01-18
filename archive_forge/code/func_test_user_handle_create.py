from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import user
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_user_handle_create(self):
    mock_user = self._get_mock_user()
    self.users.create.return_value = mock_user
    self.users.get.return_value = mock_user
    self.users.add_to_group = mock.MagicMock()
    self.assertEqual('test_user_1', self.test_user.properties.get(user.KeystoneUser.NAME))
    self.assertEqual('Test user', self.test_user.properties.get(user.KeystoneUser.DESCRIPTION))
    self.assertEqual('default', self.test_user.properties.get(user.KeystoneUser.DOMAIN))
    self.assertEqual(True, self.test_user.properties.get(user.KeystoneUser.ENABLED))
    self.assertEqual('abc@xyz.com', self.test_user.properties.get(user.KeystoneUser.EMAIL))
    self.assertEqual('password', self.test_user.properties.get(user.KeystoneUser.PASSWORD))
    self.assertEqual('project_1', self.test_user.properties.get(user.KeystoneUser.DEFAULT_PROJECT))
    self.assertEqual(['group1', 'group2'], self.test_user.properties.get(user.KeystoneUser.GROUPS))
    self.test_user.handle_create()
    self.users.create.assert_called_once_with(name='test_user_1', description='Test user', domain='default', enabled=True, email='abc@xyz.com', password='password', default_project='project_1')
    self.assertEqual(mock_user.id, self.test_user.resource_id)
    for group in ['group1', 'group2']:
        self.users.add_to_group.assert_any_call(self.test_user.resource_id, group)