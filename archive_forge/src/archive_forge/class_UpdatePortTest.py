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
class UpdatePortTest(common.HeatTestCase):
    scenarios = [('with_secgrp', dict(secgrp=['8a2f582a-e1cd-480f-b85d-b02631c10656'], name='test', value_specs={}, fixed_ips=None, addr_pair=None, vnic_type=None)), ('with_no_name', dict(secgrp=['8a2f582a-e1cd-480f-b85d-b02631c10656'], orig_name='original', name=None, value_specs={}, fixed_ips=None, addr_pair=None, vnic_type=None)), ('with_empty_values', dict(secgrp=[], name='test', value_specs={}, fixed_ips=[], addr_pair=[], vnic_type=None)), ('with_fixed_ips', dict(secgrp=None, value_specs={}, fixed_ips=[{'subnet_id': 'd0e971a6-a6b4-4f4c', 'ip_address': '10.0.0.2'}], addr_pair=None, vnic_type=None)), ('with_addr_pair', dict(secgrp=None, value_specs={}, fixed_ips=None, addr_pair=[{'ip_address': '10.0.3.21', 'mac_address': '00-B0-D0-86'}], vnic_type=None)), ('with_value_specs', dict(secgrp=None, value_specs={'binding:vnic_type': 'direct'}, fixed_ips=None, addr_pair=None, vnic_type=None)), ('normal_vnic', dict(secgrp=None, value_specs={}, fixed_ips=None, addr_pair=None, vnic_type='normal')), ('direct_vnic', dict(secgrp=None, value_specs={}, fixed_ips=None, addr_pair=None, vnic_type='direct')), ('physical_direct_vnic', dict(secgrp=None, value_specs={}, fixed_ips=None, addr_pair=None, vnic_type='direct-physical')), ('baremetal_vnic', dict(secgrp=None, value_specs={}, fixed_ips=None, addr_pair=None, vnic_type='baremetal')), ('virtio_forwarder_vnic', dict(secgrp=None, value_specs={}, fixed_ips=None, addr_pair=None, vnic_type='virtio-forwarder')), ('smart_nic_vnic', dict(secgrp=None, value_specs={}, fixed_ips=None, addr_pair=None, vnic_type='smart-nic')), ('with_all', dict(secgrp=['8a2f582a-e1cd-480f-b85d-b02631c10656'], value_specs={}, fixed_ips=[{'subnet_id': 'd0e971a6-a6b4-4f4c', 'ip_address': '10.0.0.2'}], addr_pair=[{'ip_address': '10.0.3.21', 'mac_address': '00-B0-D0-86-BB-F7'}], vnic_type='normal'))]

    def test_update_port(self):
        t = template_format.parse(neutron_port_template)
        create_name = getattr(self, 'orig_name', None)
        if create_name is not None:
            t['resources']['port']['properties']['name'] = create_name
        stack = utils.parse_stack(t)

        def res_id(client, resource, name_or_id, cmd_resource=None):
            return {'network': 'net1234', 'subnet': 'sub1234'}[resource]
        self.patchobject(neutronV20, 'find_resourceid_by_name_or_id', side_effect=res_id)
        create_port_result = {'port': {'status': 'BUILD', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
        show_port_result = {'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'fixed_ips': {'subnet_id': 'd0e971a6-a6b4-4f4c-8c88-b75e9c120b7e', 'ip_address': '10.0.0.2'}}}
        create_port = self.patchobject(neutronclient.Client, 'create_port', return_value=create_port_result)
        update_port = self.patchobject(neutronclient.Client, 'update_port')
        show_port = self.patchobject(neutronclient.Client, 'show_port', return_value=show_port_result)
        fake_groups_list = {'security_groups': [{'tenant_id': 'dc4b074874244f7693dd65583733a758', 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3', 'name': 'default', 'security_group_rules': [], 'description': 'no protocol'}]}
        self.patchobject(neutronclient.Client, 'list_security_groups', return_value=fake_groups_list)
        set_tag_mock = self.patchobject(neutronclient.Client, 'replace_tag')
        props = {'network_id': u'net1234', 'fixed_ips': [{'subnet_id': 'sub1234', 'ip_address': '10.0.3.21'}], 'name': create_name if create_name is not None else utils.PhysName(stack.name, 'port'), 'admin_state_up': True, 'device_owner': u'network:dhcp', 'device_id': '', 'binding:vnic_type': 'normal'}
        update_props = props.copy()
        update_props['name'] = getattr(self, 'name', create_name)
        update_props['security_groups'] = self.secgrp
        update_props['value_specs'] = self.value_specs
        update_props['tags'] = ['test_tag']
        if self.fixed_ips:
            update_props['fixed_ips'] = self.fixed_ips
        update_props['allowed_address_pairs'] = self.addr_pair
        update_props['binding:vnic_type'] = self.vnic_type
        update_dict = update_props.copy()
        if update_props['allowed_address_pairs'] is None:
            update_dict['allowed_address_pairs'] = []
        if update_props['security_groups'] is None:
            update_dict['security_groups'] = ['0389f747-7785-4757-b7bb-2ab07e4b09c3']
        if update_props['name'] is None:
            update_dict['name'] = utils.PhysName(stack.name, 'port')
        value_specs = update_dict.pop('value_specs')
        if value_specs:
            for value_spec in value_specs.items():
                update_dict[value_spec[0]] = value_spec[1]
        tags = update_dict.pop('tags')
        port = stack['port']
        self.assertIsNone(scheduler.TaskRunner(port.create)())
        create_port.assert_called_once_with({'port': props})
        update_snippet = rsrc_defn.ResourceDefinition(port.name, port.type(), update_props)
        self.assertIsNone(scheduler.TaskRunner(port.handle_update, update_snippet, {}, update_props)())
        update_port.assert_called_once_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'port': update_dict})
        set_tag_mock.assert_called_with('ports', port.resource_id, {'tags': tags})
        create_snippet = rsrc_defn.ResourceDefinition(port.name, port.type(), props)
        after_props, before_props = port._prepare_update_props(update_snippet, create_snippet)
        self.assertIsNotNone(port.update_template_diff_properties(after_props, before_props))
        scheduler.TaskRunner(port.handle_update, update_snippet, {}, {'fixed_ips': None})()
        scheduler.TaskRunner(port.handle_update, update_snippet, {}, {})()
        self.assertEqual(1, update_port.call_count)
        show_port.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')