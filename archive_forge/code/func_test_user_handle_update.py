from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import user
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_user_handle_update(self):
    self.test_user.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {user.KeystoneUser.NAME: 'test_user_1_updated', user.KeystoneUser.DESCRIPTION: 'Test User updated', user.KeystoneUser.ENABLED: False, user.KeystoneUser.EMAIL: 'xyz@abc.com', user.KeystoneUser.PASSWORD: 'passWORD', user.KeystoneUser.DEFAULT_PROJECT: 'project_2', user.KeystoneUser.GROUPS: ['group1', 'group3']}
    self.test_user.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.users.update.assert_called_once_with(user=self.test_user.resource_id, domain=self.test_user.properties[user.KeystoneUser.DOMAIN], name=prop_diff[user.KeystoneUser.NAME], description=prop_diff[user.KeystoneUser.DESCRIPTION], email=prop_diff[user.KeystoneUser.EMAIL], password=prop_diff[user.KeystoneUser.PASSWORD], default_project=prop_diff[user.KeystoneUser.DEFAULT_PROJECT], enabled=prop_diff[user.KeystoneUser.ENABLED])
    for group in ['group3']:
        self.users.add_to_group.assert_any_call(self.test_user.resource_id, group)
    for group in ['group2']:
        self.users.remove_from_group.assert_any_call(self.test_user.resource_id, group)
    self.roles = self.keystoneclient.roles
    self.roles.revoke.assert_not_called()
    self.roles.grant.assert_not_called()