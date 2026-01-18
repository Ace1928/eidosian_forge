from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import user
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_user_handle_delete(self):
    self.test_user.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    self.users.delete.return_value = None
    self.assertIsNone(self.test_user.handle_delete())
    self.users.delete.assert_called_once_with(self.test_user.resource_id)
    for group in ['group1', 'group2']:
        self.users.remove_from_group.assert_any_call(self.test_user.resource_id, group)