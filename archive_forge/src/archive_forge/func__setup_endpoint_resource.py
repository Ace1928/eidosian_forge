import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import endpoint
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _setup_endpoint_resource(self, stack_name, use_default=False):
    tmpl_data = copy.deepcopy(keystone_endpoint_template)
    if use_default:
        props = tmpl_data['resources']['test_endpoint']['properties']
        del props['name']
        del props['enabled']
    test_stack = stack.Stack(self.ctx, stack_name, template.Template(tmpl_data))
    r_endpoint = test_stack['test_endpoint']
    r_endpoint.client = mock.MagicMock()
    r_endpoint.client.return_value = self.keystoneclient
    r_endpoint.client_plugin = mock.MagicMock()
    r_endpoint.client_plugin.return_value = self.keystone_client_plugin
    return r_endpoint