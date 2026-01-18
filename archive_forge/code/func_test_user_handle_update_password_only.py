from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import user
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_user_handle_update_password_only(self):
    self.test_user.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {user.KeystoneUser.PASSWORD: 'passWORD'}
    self.test_user.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.users.update.assert_called_once_with(user=self.test_user.resource_id, domain=self.test_user.properties[user.KeystoneUser.DOMAIN], password=prop_diff[user.KeystoneUser.PASSWORD])
    self.users.add_to_group.assert_not_called()
    self.users.remove_from_group.assert_not_called()