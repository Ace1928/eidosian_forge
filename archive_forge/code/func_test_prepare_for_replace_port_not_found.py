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
def test_prepare_for_replace_port_not_found(self):
    t = template_format.parse(neutron_port_template)
    stack = utils.parse_stack(t)
    port = stack['port']
    port.resource_id = 'test_res_id'
    port._show_resource = mock.Mock(side_effect=qe.NotFound)
    port.data_set = mock.Mock()
    n_client = mock.Mock()
    port.client = mock.Mock(return_value=n_client)
    port.prepare_for_replace()
    self.assertTrue(port._show_resource.called)
    self.assertFalse(port.data_set.called)
    self.assertFalse(n_client.update_port.called)