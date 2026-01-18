from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.engine.cfn import template as cfntemplate
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import template as hottemplate
from heat.engine import resource as res
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_stack_update_enabled_convergence_engine(self):
    stack_name = 'service_update_test_stack'
    params = {'foo': 'bar'}
    template = '{ "Template": "data" }'
    old_stack = tools.get_stack(stack_name, self.ctx, template=tools.string_template_five, convergence=True)
    old_stack.timeout_mins = 1
    old_stack.store()
    stack = tools.get_stack(stack_name, self.ctx, template=tools.string_template_five_update, convergence=True)
    self._stub_update_mocks(old_stack)
    templatem.Template.return_value = stack.t
    environment.Environment.return_value = stack.env
    parser.Stack.return_value = stack
    self.patchobject(stack, 'validate', return_value=None)
    api_args = {'timeout_mins': 60, 'disable_rollback': False, rpc_api.PARAM_CONVERGE: False}
    result = self.man.update_stack(self.ctx, old_stack.identifier(), template, params, None, api_args)
    self.assertTrue(old_stack.convergence)
    self.assertEqual(old_stack.identifier(), result)
    self.assertIsInstance(result, dict)
    self.assertTrue(result['stack_id'])
    parser.Stack.load.assert_called_once_with(self.ctx, stack=mock.ANY, check_refresh_cred=True)
    templatem.Template.assert_called_once_with(template, files=None)
    environment.Environment.assert_called_once_with(params)