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
def test_property_roles_validate_schema(self):
    schema = MixinClass.mixin_properties_schema[MixinClass.ROLES]
    self.assertEqual(True, schema.update_allowed, 'update_allowed for property %s is modified' % MixinClass.ROLES)
    self.assertEqual(properties.Schema.LIST, schema.type, 'type for property %s is modified' % MixinClass.ROLES)
    self.assertEqual('List of role assignments.', schema.description, 'description for property %s is modified' % MixinClass.ROLES)