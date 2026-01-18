import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import role
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_role_handle_update(self):
    test_role = self._get_rsrc()
    test_role.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {role.KeystoneRole.NAME: 'test_role_1_updated'}
    test_role.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.roles.update.assert_called_once_with(role=test_role.resource_id, name=prop_diff[role.KeystoneRole.NAME])