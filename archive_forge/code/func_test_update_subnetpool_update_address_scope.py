from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import subnetpool
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.neutron import inline_templates
from heat.tests import utils
def test_update_subnetpool_update_address_scope(self):
    update_subnetpool = self.patchobject(neutronclient.Client, 'update_subnetpool')
    rsrc = self.create_subnetpool()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', ref_id)
    props = {'name': 'the_new_sp', 'address_scope': 'new_test', 'prefixes': ['10.0.0.0/8', '10.6.0.0/16']}
    update_dict = {'name': 'the_new_sp', 'address_scope_id': 'new_test', 'prefixes': ['10.0.0.0/8', '10.6.0.0/16']}
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    self.assertIsNone(rsrc.handle_update(update_snippet, {}, props))
    self.assertEqual(3, self.find_resource.call_count)
    update_subnetpool.assert_called_once_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'subnetpool': update_dict})