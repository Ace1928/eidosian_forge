import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import openstacksdk
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
def test_update_subnet_with_no_name(self):
    stack_name = utils.random_name()
    update_props = {'subnet': {'name': None}}
    update_props_name = {'subnet': {'name': utils.PhysName(stack_name, 'test_subnet')}}
    t, stack = self._setup_mock(stack_name)
    self.patchobject(stack['net'], 'FnGetRefId', return_value='fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    rsrc = self.create_subnet(t, stack, 'sub_net')
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    rsrc.validate()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('91e47a57-7508-46fe-afc9-fc454e8580e1', ref_id)
    self.assertIsNone(rsrc.FnGetAtt('network_id'))
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', rsrc.FnGetAtt('network_id'))
    self.assertEqual('8.8.8.8', rsrc.FnGetAtt('dns_nameservers')[0])
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), update_props['subnet'])
    rsrc.handle_update(update_snippet, {}, update_props['subnet'])
    self.update_mock.assert_called_once_with('91e47a57-7508-46fe-afc9-fc454e8580e1', update_props_name)
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())
    rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE, 'to delete again')
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())