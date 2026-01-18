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
def test_property_description_validate_schema(self):
    schema = group.KeystoneGroup.properties_schema[group.KeystoneGroup.DESCRIPTION]
    self.assertEqual(True, schema.update_allowed, 'update_allowed for property %s is modified' % group.KeystoneGroup.DESCRIPTION)
    self.assertEqual(properties.Schema.STRING, schema.type, 'type for property %s is modified' % group.KeystoneGroup.DESCRIPTION)
    self.assertEqual('Description of keystone group.', schema.description, 'description for property %s is modified' % group.KeystoneGroup.DESCRIPTION)
    self.assertEqual('', schema.default, 'default for property %s is modified' % group.KeystoneGroup.DESCRIPTION)