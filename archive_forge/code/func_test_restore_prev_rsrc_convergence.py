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
def test_restore_prev_rsrc_convergence(self):
    t = template_format.parse(neutron_port_template)
    stack = utils.parse_stack(t)
    stack.store()
    prev_rsrc = stack['port']
    prev_rsrc.resource_id = 'prev-rsrc'
    prev_rsrc.state_set(prev_rsrc.UPDATE, prev_rsrc.COMPLETE)
    existing_rsrc = stack['port']
    existing_rsrc.current_template_id = stack.t.id
    existing_rsrc.resource_id = 'existing-rsrc'
    existing_rsrc.state_set(existing_rsrc.UPDATE, existing_rsrc.COMPLETE)
    prev_rsrc.replaced_by = existing_rsrc.id
    _value = {'subnet_id': 'test_subnet', 'ip_address': '42.42.42.42'}
    prev_rsrc._data = {'port_fip': jsonutils.dumps(_value)}
    n_client = mock.Mock()
    prev_rsrc.client = mock.Mock(return_value=n_client)
    prev_rsrc.restore_prev_rsrc(convergence=True)
    expected_existing_props = {'port': {'fixed_ips': []}}
    expected_prev_props = {'port': {'fixed_ips': _value}}
    n_client.update_port.assert_has_calls([mock.call(existing_rsrc.resource_id, expected_existing_props), mock.call(prev_rsrc.resource_id, expected_prev_props)])