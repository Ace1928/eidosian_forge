from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_port_needs_update(self):
    t = template_format.parse(neutron_port_template)
    t['resources']['port']['properties'].pop('fixed_ips')
    stack = utils.parse_stack(t)
    props = {'network_id': u'net1234', 'name': utils.PhysName(stack.name, 'port'), 'admin_state_up': True, 'device_owner': u'network:dhcp', 'device_id': '', 'binding:vnic_type': 'normal'}
    self.find_mock.return_value = 'net1234'
    self.create_mock.return_value = {'port': {'status': 'BUILD', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.port_show_mock.return_value = {'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'fixed_ips': {'subnet_id': 'd0e971a6-a6b4-4f4c-8c88-b75e9c120b7e', 'ip_address': '10.0.0.2'}}}
    port = stack['port']
    scheduler.TaskRunner(port.create)()
    self.create_mock.assert_called_once_with({'port': props})
    new_props = props.copy()
    new_props['replacement_policy'] = 'REPLACE_ALWAYS'
    new_props['network'] = new_props.pop('network_id')
    update_snippet = rsrc_defn.ResourceDefinition(port.name, port.type(), new_props)
    self.assertRaises(resource.UpdateReplace, port._needs_update, update_snippet, port.frozen_definition(), new_props, port.properties, None)
    new_props['replacement_policy'] = 'AUTO'
    update_snippet = rsrc_defn.ResourceDefinition(port.name, port.type(), new_props)
    self.assertTrue(port._needs_update(update_snippet, port.frozen_definition(), new_props, port.properties, None))