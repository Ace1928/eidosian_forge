import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_router(self):
    t = template_format.parse(neutron_template)
    tags = ['for_test']
    t['resources']['router']['properties']['tags'] = tags
    stack = utils.parse_stack(t)
    create_body = {'router': {'name': utils.PhysName(stack.name, 'router'), 'availability_zone_hints': ['az1'], 'admin_state_up': True}}
    router_base_info = {'router': {'status': 'BUILD', 'external_gateway_info': None, 'name': utils.PhysName(stack.name, 'router'), 'admin_state_up': True, 'tenant_id': '3e21026f2dc94372b105808c0e721661', 'id': '3e46229d-8fce-4733-819a-b5fe630550f8'}}
    router_active_info = copy.deepcopy(router_base_info)
    router_active_info['router']['status'] = 'ACTIVE'
    self.create_mock.return_value = router_base_info
    self.show_mock.side_effect = [router_base_info, router_active_info, qe.NeutronClientException(status_code=404), router_active_info, qe.NeutronClientException(status_code=404)]
    agents_info = {'agents': [{'admin_state_up': True, 'agent_type': 'L3 agent', 'alive': True, 'binary': 'neutron-l3-agent', 'configurations': {'ex_gw_ports': 1, 'floating_ips': 0, 'gateway_external_network_id': '', 'handle_internal_only_routers': True, 'interface_driver': 'DummyDriver', 'interfaces': 1, 'router_id': '', 'routers': 1, 'use_namespaces': True}, 'created_at': '2014-03-11 05:00:05', 'description': None, 'heartbeat_timestamp': '2014-03-11 05:01:49', 'host': 'l3_agent_host', 'id': '792ff887-6c85-4a56-b518-23f24fa65581', 'started_at': '2014-03-11 05:00:05', 'topic': 'l3_agent'}]}
    agents_info1 = copy.deepcopy(agents_info)
    agent = agents_info1['agents'][0]
    agent['id'] = '63b3fd83-2c5f-4dad-b3ae-e0f83a40f216'
    self.list_l3_hr_mock.side_effect = [{'agents': []}, agents_info, agents_info1]
    self.delete_mock.side_effect = [None, qe.NeutronClientException(status_code=404)]
    set_tag_mock = self.patchobject(neutronclient.Client, 'replace_tag')
    rsrc = self.create_router(t, stack, 'router')
    self.create_mock.assert_called_with(create_body)
    set_tag_mock.assert_called_with('routers', rsrc.resource_id, {'tags': tags})
    rsrc.validate()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('3e46229d-8fce-4733-819a-b5fe630550f8', ref_id)
    self.assertIsNone(rsrc.FnGetAtt('tenant_id'))
    self.assertEqual('3e21026f2dc94372b105808c0e721661', rsrc.FnGetAtt('tenant_id'))
    prop_diff = {'admin_state_up': False, 'name': 'myrouter', 'l3_agent_ids': ['63b3fd83-2c5f-4dad-b3ae-e0f83a40f216'], 'tags': ['new_tag']}
    props = copy.copy(rsrc.properties.data)
    props.update(prop_diff)
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    rsrc.handle_update(update_snippet, {}, prop_diff)
    set_tag_mock.assert_called_with('routers', rsrc.resource_id, {'tags': ['new_tag']})
    self.update_mock.assert_called_with('3e46229d-8fce-4733-819a-b5fe630550f8', {'router': {'name': 'myrouter', 'admin_state_up': False}})
    prop_diff = {'l3_agent_ids': ['4c692423-2c5f-4dad-b3ae-e2339f58539f', '8363b3fd-2c5f-4dad-b3ae-0f216e0f83a4']}
    props = copy.copy(rsrc.properties.data)
    props.update(prop_diff)
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    rsrc.handle_update(update_snippet, {}, prop_diff)
    add_router_calls = [mock.call(u'792ff887-6c85-4a56-b518-23f24fa65581', {'router_id': u'3e46229d-8fce-4733-819a-b5fe630550f8'}), mock.call(u'63b3fd83-2c5f-4dad-b3ae-e0f83a40f216', {'router_id': u'3e46229d-8fce-4733-819a-b5fe630550f8'}), mock.call(u'4c692423-2c5f-4dad-b3ae-e2339f58539f', {'router_id': u'3e46229d-8fce-4733-819a-b5fe630550f8'}), mock.call(u'8363b3fd-2c5f-4dad-b3ae-0f216e0f83a4', {'router_id': u'3e46229d-8fce-4733-819a-b5fe630550f8'})]
    remove_router_calls = [mock.call(u'792ff887-6c85-4a56-b518-23f24fa65581', u'3e46229d-8fce-4733-819a-b5fe630550f8'), mock.call(u'63b3fd83-2c5f-4dad-b3ae-e0f83a40f216', u'3e46229d-8fce-4733-819a-b5fe630550f8')]
    self.add_router_mock.assert_has_calls(add_router_calls)
    self.remove_router_mock.assert_has_calls(remove_router_calls)
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())
    rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE, 'to delete again')
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())