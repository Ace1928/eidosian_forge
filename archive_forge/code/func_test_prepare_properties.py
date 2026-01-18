from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine import attributes
from heat.engine import properties
from heat.engine.resources.openstack.neutron import net
from heat.engine.resources.openstack.neutron import neutron as nr
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_prepare_properties(self):
    data = {'admin_state_up': False, 'value_specs': {'router:external': True}}
    p = properties.Properties(net.Net.properties_schema, data)
    props = nr.NeutronResource.prepare_properties(p, 'resource_name')
    self.assertEqual({'name': 'resource_name', 'router:external': True, 'admin_state_up': False, 'shared': False}, props)