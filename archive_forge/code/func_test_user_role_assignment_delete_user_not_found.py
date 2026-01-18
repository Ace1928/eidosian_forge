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
def test_user_role_assignment_delete_user_not_found(self):
    self.keystone_client_plugin.get_user_id.side_effect = [exception.EntityNotFound]
    self.assertIsNone(self.test_role_assignment.handle_delete())
    self.roles.revoke.assert_not_called()