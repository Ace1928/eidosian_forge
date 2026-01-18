from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import group
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_property_domain_validate_schema(self):
    schema = group.KeystoneGroup.properties_schema[group.KeystoneGroup.DOMAIN]
    self.assertEqual(True, schema.update_allowed, 'update_allowed for property %s is modified' % group.KeystoneGroup.DOMAIN)
    self.assertEqual(properties.Schema.STRING, schema.type, 'type for property %s is modified' % group.KeystoneGroup.DOMAIN)
    self.assertEqual('Name or id of keystone domain.', schema.description, 'description for property %s is modified' % group.KeystoneGroup.DOMAIN)
    self.assertEqual([constraints.CustomConstraint('keystone.domain')], schema.constraints, 'constrains for property %s is modified' % group.KeystoneGroup.DOMAIN)
    self.assertEqual('default', schema.default, 'default for property %s is modified' % group.KeystoneGroup.DOMAIN)