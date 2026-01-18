import copy
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import trunk
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests import utils
from neutronclient.common import exceptions as ncex
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
def test_update_subport_add(self):
    t = template_format.parse(update_template)
    stack = utils.parse_stack(t)
    rsrc_defn = stack.defn.resource_definition('trunk')
    rsrc = trunk.Trunk('trunk', rsrc_defn, stack)
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    props = copy.deepcopy(t['resources']['trunk']['properties'])
    props['sub_ports'].append({'port': 'subport_3_id', 'segmentation_type': 'vlan', 'segmentation_id': 103})
    rsrc_defn = rsrc_defn.freeze(properties=props)
    scheduler.TaskRunner(rsrc.update, rsrc_defn)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    self.update_trunk_mock.assert_not_called()
    self.trunk_remove_subports_mock.assert_not_called()
    self.trunk_add_subports_mock.assert_called_once_with('trunk id', {'sub_ports': [{'port_id': 'subport_3_id', 'segmentation_id': 103, 'segmentation_type': 'vlan'}]})