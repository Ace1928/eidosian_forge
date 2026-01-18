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
def test_properties_title(self):
    property_title_map = {MixinClass.ROLES: 'roles'}
    for actual_title, expected_title in property_title_map.items():
        self.assertEqual(expected_title, actual_title, 'KeystoneRoleAssignmentMixin PROPERTIES(%s) title modified.' % actual_title)