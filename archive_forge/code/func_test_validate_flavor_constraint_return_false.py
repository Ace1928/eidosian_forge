from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_flavor_constraint_return_false(self):
    self.t['resources']['node-group']['properties'].pop('floating_ip_pool')
    self.t['resources']['node-group']['properties'].pop('volume_type')
    ngt = self._init_ngt(self.t)
    self.patchobject(nova.FlavorConstraint, 'validate').return_value = False
    ex = self.assertRaises(exception.StackValidationFailed, ngt.validate)
    self.assertEqual(u"Property error: resources.node-group.properties.flavor: Error validating value 'm1.large'", str(ex))