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
def test_validate_1(self):
    self.test_role_assignment.properties = mock.MagicMock()
    self.test_role_assignment.properties.get.return_value = [dict(role='role1')]
    self.assertRaises(exception.StackValidationFailed, self.test_role_assignment.validate)