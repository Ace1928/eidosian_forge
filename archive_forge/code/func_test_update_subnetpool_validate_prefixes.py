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
def test_update_subnetpool_validate_prefixes(self):
    update_subnetpool = self.patchobject(neutronclient.Client, 'update_subnetpool')
    rsrc = self.create_subnetpool()
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', ref_id)
    prefix_old = rsrc.properties['prefixes']
    props = {'name': 'the_new_sp', 'prefixes': ['10.5.0.0/16']}
    prefix_new = props['prefixes']
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    errMessage = 'Property prefixes updated value %(value1)s should be superset of existing value %(value2)s.' % dict(value1=sorted(prefix_new), value2=sorted(prefix_old))
    error = self.assertRaises(exception.StackValidationFailed, rsrc.handle_update, update_snippet, {}, props)
    self.assertEqual(errMessage, str(error))
    update_subnetpool.assert_not_called()
    props = {'name': 'the_new_sp', 'prefixes': ['10.0.0.0/8', '10.6.0.0/16']}
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    self.assertIsNone(rsrc.handle_update(update_snippet, {}, props))
    update_subnetpool.assert_called_once_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'subnetpool': props})