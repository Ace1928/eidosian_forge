import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import role_assignments
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
def test_role_assignment_update_roles_no_change(self):
    prop_diff = {}
    self.test_role_assignment.update_assignment(group_id='group_1', prop_diff=prop_diff)
    self.assertEqual(0, self.roles.grant.call_count)
    self.assertEqual(0, self.roles.revoke.call_count)
    self.test_role_assignment.update_assignment(user_id='user_1', prop_diff=prop_diff)
    self.assertEqual(0, self.roles.grant.call_count)
    self.assertEqual(0, self.roles.revoke.call_count)