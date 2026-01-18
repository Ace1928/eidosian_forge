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
def test_port_needs_update_network(self):
    net1 = '9cfe6c74-c105-4906-9a1f-81d9064e9bca'
    net2 = '0064eec9-5681-4ba7-a745-6f8e32db9503'
    props = {'network_id': net1, 'name': 'test_port', 'device_owner': u'network:dhcp', 'binding:vnic_type': 'normal', 'device_id': ''}
    create_kwargs = props.copy()
    create_kwargs['admin_state_up'] = True
    self.find_mock.side_effect = [net1] * 8 + [net2] * 2 + [net1]
    self.create_mock.return_value = {'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.port_show_mock.return_value = {'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'fixed_ips': {'subnet_id': 'd0e971a6-a6b4-4f4c-8c88-b75e9c120b7e', 'ip_address': '10.0.0.2'}}}
    tmpl = neutron_port_template.replace('network: net1234', 'network_id: 9cfe6c74-c105-4906-9a1f-81d9064e9bca')
    t = template_format.parse(tmpl)
    t['resources']['port']['properties'].pop('fixed_ips')
    t['resources']['port']['properties']['name'] = 'test_port'
    stack = utils.parse_stack(t)
    port = stack['port']
    scheduler.TaskRunner(port.create)()
    self.assertEqual((port.CREATE, port.COMPLETE), port.state)
    self.create_mock.assert_called_once_with({'port': create_kwargs})
    new_props = props.copy()
    new_props['network'] = new_props.pop('network_id')
    update_snippet = rsrc_defn.ResourceDefinition(port.name, port.type(), new_props)
    scheduler.TaskRunner(port.update, update_snippet)()
    self.assertEqual((port.UPDATE, port.COMPLETE), port.state)
    self.assertEqual(0, self.update_mock.call_count)
    new_props['network'] = 'net1234'
    update_snippet = rsrc_defn.ResourceDefinition(port.name, port.type(), new_props)
    scheduler.TaskRunner(port.update, update_snippet)()
    self.assertEqual((port.UPDATE, port.COMPLETE), port.state)
    self.assertEqual(0, self.update_mock.call_count)
    new_props['network'] = 'net5678'
    update_snippet = rsrc_defn.ResourceDefinition(port.name, port.type(), new_props)
    updater = scheduler.TaskRunner(port.update, update_snippet)
    self.assertRaises(resource.UpdateReplace, updater)
    self.assertEqual(11, self.find_mock.call_count)